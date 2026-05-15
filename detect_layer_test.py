# steps/detect_layer_2.py
"""
Step 3 -- Layer 2: Join the two dining-area ducts into one connected
3-segment path (horizontal -> vertical -> horizontal) -- a clean U-shape.

Strategy:
  Phase 1: Walk LEFT from the top duct's left endpoint, jumping by
           ~half-duct-diameter at a time. At each jump, cast vertical
           rays to find the top and bottom walls. Compare wall y values
           to the seed values; once they drift (sign tells us turn
           direction), we've found the first elbow.
  Phase 2: From corner 1, walk vertically in the detected turn direction.
           At each jump, cast horizontal rays to find left/right walls.
           Once they drift, we've found the second elbow.
  Phase 3: Snap a horizontal segment from corner 2 to the bottom duct's
           left endpoint.

Coordinates: OpenCV convention. (0,0) at top-left; y increases downward.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# KNOBS
# =============================================================================
# How much wall-position drift to ignore as noise (pixels).
# Anything beyond this counts as an elbow.
WALL_NOISE_TOLERANCE_PX = 5

# Max distance a perpendicular ray will look for a wall
WALL_SEARCH_MAX_PX = 200

# Safety caps (number of jumps before bailing out of a phase)
MAX_HORIZONTAL_JUMPS = 60
MAX_VERTICAL_JUMPS   = 60

# How far past the Layer 1 endpoint to jump before starting Phase 1.
# Needs to clear the fitting at the Layer 1 stop point and re-seed walls
# inside the clean horizontal section. 0 = no initial jump (use Layer 1 seed).
INITIAL_JUMP_PX = 70

# Debug overlay PNG (rays + wall hits + corners marked)
DEBUG_OVERLAY = True

# Drawing
LINE_COLOR  = (255, 0, 0)   # blue (BGR)
LINE_THICK  = 8
DOT_RADIUS  = 6
DBG_SAMPLE  = (0, 255, 255)   # yellow
DBG_RAY     = (0, 200, 0)     # green
DBG_WALL    = (0, 0, 255)     # red
DBG_CORNER  = (255, 0, 255)   # magenta
# =============================================================================


def binarize(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    return binary


# ----- Ray casting ---------------------------------------------------------

def ray_up(binary, x, y, max_dist=WALL_SEARCH_MAX_PX):
    """From (x,y) walk in -y direction; return wall y or None."""
    h, w = binary.shape
    for d in range(1, max_dist + 1):
        yy = y - d
        if yy < 0 or x < 0 or x >= w:
            return None
        if binary[yy, x] > 0:
            return yy
    return None


def ray_down(binary, x, y, max_dist=WALL_SEARCH_MAX_PX):
    h, w = binary.shape
    for d in range(1, max_dist + 1):
        yy = y + d
        if yy >= h or x < 0 or x >= w:
            return None
        if binary[yy, x] > 0:
            return yy
    return None


def ray_left(binary, x, y, max_dist=WALL_SEARCH_MAX_PX):
    h, w = binary.shape
    for d in range(1, max_dist + 1):
        xx = x - d
        if xx < 0 or y < 0 or y >= h:
            return None
        if binary[y, xx] > 0:
            return xx
    return None


def ray_right(binary, x, y, max_dist=WALL_SEARCH_MAX_PX):
    h, w = binary.shape
    for d in range(1, max_dist + 1):
        xx = x + d
        if xx >= w or y < 0 or y >= h:
            return None
        if binary[y, xx] > 0:
            return xx
    return None


# ----- Phase 1: horizontal walk to find first elbow ------------------------

def walk_horizontal(binary, start_x, start_y, jump_px,
                    seed_top_y, seed_bottom_y, direction=-1,
                    debug_canvas=None):
    """
    Walk horizontally from (start_x, start_y) in `direction` (-1 left, +1 right)
    until walls drift (elbow).
    Returns dict:
      - corner_x:        x at last good (pre-drift) position
      - corner_y:        start_y (still on horizontal centerline)
      - turn_dir:        +1 (turn down, y increases) or -1 (turn up, y decreases) or 0 (none / dead end)
      - reason:          'elbow' | 'no_walls' | 'max_jumps' | 'bad_initial_jump'
      - samples:         list of (x, top_y, bottom_y) for debug
      - seed_top_y:      wall y used as seed (may be re-measured after initial jump)
      - seed_bottom_y:   ditto
    """
    assert direction in (-1, +1), "direction must be -1 (left) or +1 (right)"
    samples = []

    # ---- Initial jump past the fitting at the Layer 1 endpoint ----
    if INITIAL_JUMP_PX > 0:
        x0 = start_x + direction * INITIAL_JUMP_PX
        new_top    = ray_up(binary,   x0, start_y)
        new_bottom = ray_down(binary, x0, start_y)
        samples.append((x0, new_top, new_bottom))

        if debug_canvas is not None:
            # Mark the initial jump sample distinctly (cyan)
            cv2.circle(debug_canvas, (x0, start_y), 5, (255, 255, 0), -1)
            cv2.line(debug_canvas, (x0, start_y),
                     (x0, start_y - WALL_SEARCH_MAX_PX), DBG_RAY, 1)
            cv2.line(debug_canvas, (x0, start_y),
                     (x0, start_y + WALL_SEARCH_MAX_PX), DBG_RAY, 1)
            if new_top is not None:
                cv2.circle(debug_canvas, (x0, new_top), 4, DBG_WALL, -1)
            if new_bottom is not None:
                cv2.circle(debug_canvas, (x0, new_bottom), 4, DBG_WALL, -1)

        if new_top is None or new_bottom is None:
            # Initial jump landed somewhere with no walls -- can't proceed
            return {
                "corner_x": start_x,
                "corner_y": start_y,
                "turn_dir": 0,
                "reason":   "bad_initial_jump",
                "samples":  samples,
                "seed_top_y":    seed_top_y,
                "seed_bottom_y": seed_bottom_y,
            }

        # Re-seed using walls measured at the post-jump position
        seed_top_y    = new_top
        seed_bottom_y = new_bottom
        # The "start" of horizontal walking is now x0
        loop_start_x = x0
    else:
        loop_start_x = start_x

    last_good_x = loop_start_x

    for j in range(1, MAX_HORIZONTAL_JUMPS + 1):
        x = loop_start_x + direction * j * jump_px
        new_top    = ray_up(binary,   x, start_y)
        new_bottom = ray_down(binary, x, start_y)
        samples.append((x, new_top, new_bottom))

        if debug_canvas is not None:
            cv2.circle(debug_canvas, (x, start_y), 3, DBG_SAMPLE, -1)
            cv2.line(debug_canvas, (x, start_y),
                     (x, start_y - WALL_SEARCH_MAX_PX), DBG_RAY, 1)
            cv2.line(debug_canvas, (x, start_y),
                     (x, start_y + WALL_SEARCH_MAX_PX), DBG_RAY, 1)
            if new_top is not None:
                cv2.circle(debug_canvas, (x, new_top), 4, DBG_WALL, -1)
            if new_bottom is not None:
                cv2.circle(debug_canvas, (x, new_bottom), 4, DBG_WALL, -1)

        # If either ray found nothing -> walked out of duct, backtrack
        if new_top is None or new_bottom is None:
            return {
                "corner_x": last_good_x,
                "corner_y": start_y,
                "turn_dir": 0,
                "reason":   "no_walls",
                "samples":  samples,
                "seed_top_y":    seed_top_y,
                "seed_bottom_y": seed_bottom_y,
            }

        # Compute drift (signed)
        d_top    = new_top    - seed_top_y
        d_bottom = new_bottom - seed_bottom_y

        # Either wall drifting beyond noise tolerance = elbow
        elbow = (abs(d_top)    > WALL_NOISE_TOLERANCE_PX or
                 abs(d_bottom) > WALL_NOISE_TOLERANCE_PX)

        if elbow:
            # Direction: use whichever delta has larger magnitude
            drift = d_top if abs(d_top) >= abs(d_bottom) else d_bottom
            turn_dir = +1 if drift > 0 else -1
            return {
                "corner_x": last_good_x,
                "corner_y": start_y,
                "turn_dir": turn_dir,
                "reason":   "elbow",
                "samples":  samples,
                "seed_top_y":    seed_top_y,
                "seed_bottom_y": seed_bottom_y,
            }

        # Still horizontal -- this jump is "last good"
        last_good_x = x

    return {
        "corner_x": last_good_x,
        "corner_y": start_y,
        "turn_dir": 0,
        "reason":   "max_jumps",
        "samples":  samples,
        "seed_top_y":    seed_top_y,
        "seed_bottom_y": seed_bottom_y,
    }


# ----- Phase 2: vertical walk to find second elbow -------------------------

def walk_vertical(binary, corner_x, corner_y, jump_px, turn_dir,
                  debug_canvas=None):
    """
    Walk vertically from (corner_x, corner_y) in direction turn_dir
    (+1 = down/y increasing, -1 = up/y decreasing) until walls drift.
    Returns dict with corner_x/corner_y of the next elbow (single corner).
    """
    # Seed the vertical-section walls at the starting position
    seed_left  = ray_left(binary,  corner_x, corner_y)
    seed_right = ray_right(binary, corner_x, corner_y)
    samples = [(corner_x, corner_y, seed_left, seed_right)]
    last_good_y = corner_y

    if seed_left is None or seed_right is None:
        return {
            "corner_x": corner_x,
            "corner_y": last_good_y,
            "turn_dir": 0,
            "reason":   "no_seed_walls",
            "samples":  samples,
            "seed_left":  seed_left,
            "seed_right": seed_right,
        }

    if debug_canvas is not None:
        cv2.circle(debug_canvas, (seed_left,  corner_y), 4, DBG_WALL, -1)
        cv2.circle(debug_canvas, (seed_right, corner_y), 4, DBG_WALL, -1)

    for j in range(1, MAX_VERTICAL_JUMPS + 1):
        y = corner_y + j * jump_px * turn_dir
        new_left  = ray_left(binary,  corner_x, y)
        new_right = ray_right(binary, corner_x, y)
        samples.append((corner_x, y, new_left, new_right))

        if debug_canvas is not None:
            cv2.circle(debug_canvas, (corner_x, y), 3, DBG_SAMPLE, -1)
            cv2.line(debug_canvas, (corner_x, y),
                     (corner_x - WALL_SEARCH_MAX_PX, y), DBG_RAY, 1)
            cv2.line(debug_canvas, (corner_x, y),
                     (corner_x + WALL_SEARCH_MAX_PX, y), DBG_RAY, 1)
            if new_left is not None:
                cv2.circle(debug_canvas, (new_left,  y), 4, DBG_WALL, -1)
            if new_right is not None:
                cv2.circle(debug_canvas, (new_right, y), 4, DBG_WALL, -1)

        if new_left is None or new_right is None:
            return {
                "corner_x":   corner_x,
                "corner_y":   last_good_y,
                "turn_dir":   0,
                "reason":     "no_walls",
                "samples":    samples,
                "seed_left":  seed_left,
                "seed_right": seed_right,
            }

        d_left  = new_left  - seed_left
        d_right = new_right - seed_right
        elbow = (abs(d_left)  > WALL_NOISE_TOLERANCE_PX or
                 abs(d_right) > WALL_NOISE_TOLERANCE_PX)

        if elbow:
            drift = d_left if abs(d_left) >= abs(d_right) else d_right
            exit_dir = +1 if drift > 0 else -1
            return {
                "corner_x":   corner_x,
                "corner_y":   last_good_y,
                "turn_dir":   exit_dir,
                "reason":     "elbow",
                "samples":    samples,
                "seed_left":  seed_left,
                "seed_right": seed_right,
            }

        last_good_y = y

    return {
        "corner_x":   corner_x,
        "corner_y":   last_good_y,
        "turn_dir":   0,
        "reason":     "max_jumps",
        "samples":    samples,
        "seed_left":  seed_left,
        "seed_right": seed_right,
    }


# ----- Output rendering ----------------------------------------------------

def draw_layer1(overlay, ducts):
    for d in ducts:
        y, lx, rx = d["y"], d["left_x"], d["right_x"]
        cv2.line(overlay, (lx, y), (rx, y), LINE_COLOR, LINE_THICK)
        cv2.circle(overlay, (lx, y), DOT_RADIUS, LINE_COLOR, -1)
        cv2.circle(overlay, (rx, y), DOT_RADIUS, LINE_COLOR, -1)


def draw_layer2_corners(overlay, corners):
    """Draw straight blue segments between consecutive corner points."""
    for i in range(len(corners) - 1):
        cv2.line(overlay, corners[i], corners[i + 1], LINE_COLOR, LINE_THICK)
    for c in corners:
        cv2.circle(overlay, c, DOT_RADIUS, LINE_COLOR, -1)


# ----- Pair selectors ------------------------------------------------------

def select_dining_pair(ducts):
    """Two long horizontal ducts in dining area; connect via LEFT endpoints."""
    candidates = []
    for d in ducts:
        if d["length_px"] < 1500:                continue
        if not (4800 <= d["left_x"]  <= 5100):   continue
        if not (6700 <= d["right_x"] <= 6900):   continue
        if not (1400 <= d["y"]       <= 2600):   continue
        candidates.append(d)
    candidates.sort(key=lambda d: d["y"])
    if len(candidates) < 2:
        return None, None
    return candidates[0], candidates[1]


def select_scullery_pair(ducts):
    """Right-stub fragments in scullery area; connect via RIGHT endpoints."""
    candidates = []
    for d in ducts:
        if not (100 <= d["length_px"] <= 400):   continue
        if not (2400 <= d["left_x"]  <= 2700):   continue
        if not (2600 <= d["right_x"] <= 2900):   continue
        if not (1400 <= d["y"]       <= 2300):   continue
        candidates.append(d)
    candidates.sort(key=lambda d: d["y"])
    if len(candidates) < 2:
        return None, None
    return candidates[0], candidates[1]


# Pair configurations: each defines a selector + which endpoint to start from
# + which direction to walk. All other parameters are shared module-level knobs.
PAIRS = [
    {
        "name":      "dining",
        "selector":  select_dining_pair,
        "endpoint":  "left",   # use left_x of each duct
        "direction": -1,       # walk leftward
    },
    {
        "name":      "scullery",
        "selector":  select_scullery_pair,
        "endpoint":  "right",  # use right_x of each duct
        "direction": +1,       # walk rightward
    },
]


def trace_pair(binary, top, bottom, endpoint, direction, debug_canvas=None):
    """
    Run Phase 1 -> 2 -> 3 for one pair of ducts. Returns:
      - corners: [start, corner1, corner2_snapped, target]
      - p1, p2 dicts (or p2=None if Phase 1 found no elbow)
    """
    # Pick the endpoint x for each duct
    if endpoint == "left":
        start_x_top    = top["left_x"]
        start_x_bottom = bottom["left_x"]
    elif endpoint == "right":
        start_x_top    = top["right_x"]
        start_x_bottom = bottom["right_x"]
    else:
        raise ValueError(f"endpoint must be 'left' or 'right', got {endpoint!r}")

    start_xy  = (start_x_top,    top["y"])
    target_xy = (start_x_bottom, bottom["y"])
    jump_px   = max(10, top["height_px"])

    if debug_canvas is not None:
        draw_layer1(debug_canvas, [top, bottom])
        cv2.circle(debug_canvas, target_xy, 14, (0, 0, 255), 3)

    print(f"\nPhase 1: walking {'left' if direction<0 else 'right'} from "
          f"{start_xy}...")
    p1 = walk_horizontal(
        binary, start_x=start_x_top, start_y=top["y"], jump_px=jump_px,
        seed_top_y=top["top_y"], seed_bottom_y=top["bottom_y"],
        direction=direction,
        debug_canvas=debug_canvas,
    )
    corner1 = (p1["corner_x"], p1["corner_y"])
    print(f"  reason: {p1['reason']}, turn_dir: {p1['turn_dir']}, "
          f"corner1={corner1}")
    print(f"  re-seeded walls: top={p1.get('seed_top_y')}, "
          f"bottom={p1.get('seed_bottom_y')}")

    if p1["turn_dir"] == 0:
        print("  Phase 1 did not detect an elbow; snapping straight to target.")
        corner2 = corner1
        p2 = None
    else:
        print(f"Phase 2: walking {'down' if p1['turn_dir']>0 else 'up'} "
              f"from corner1...")
        p2 = walk_vertical(
            binary, corner_x=corner1[0], corner_y=corner1[1],
            jump_px=jump_px, turn_dir=p1["turn_dir"],
            debug_canvas=debug_canvas,
        )
        corner2 = (p2["corner_x"], p2["corner_y"])
        print(f"  reason: {p2['reason']}, turn_dir: {p2['turn_dir']}, "
              f"corner2={corner2}")

    corner2_snapped = (corner2[0], target_xy[1])
    corners = [start_xy, corner1, corner2_snapped, target_xy]
    print(f"Final corners: {corners}")

    if debug_canvas is not None:
        for c in corners:
            cv2.circle(debug_canvas, c, 8, DBG_CORNER, 2)

    return corners, p1, p2


# ----- Main ----------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "steps"))
    from load import load_and_crop, find_input_pdf

    output_dir = project_root / "output"
    pdf_in = find_input_pdf(project_root / "input")

    print(f"Loading & cropping {pdf_in.name}...")
    doc, page, image, zoom, crop_box = load_and_crop(pdf_in, dpi=300)
    doc.close()

    binary = binarize(image)

    layer1_path = output_dir / f"{pdf_in.stem}_step3_layer1.json"
    if not layer1_path.exists():
        print(f"  ERROR: Layer 1 output not found: {layer1_path}")
        sys.exit(1)
    with open(layer1_path) as f:
        layer1_ducts = json.load(f)
    print(f"  Loaded {len(layer1_ducts)} Layer 1 ducts")

    debug_canvas = image.copy() if DEBUG_OVERLAY else None
    overlay = image.copy()
    draw_layer1(overlay, layer1_ducts)

    pair_results = []
    for cfg in PAIRS:
        print(f"\n========== Pair: {cfg['name']} ==========")
        top, bottom = cfg["selector"](layer1_ducts)
        if top is None or bottom is None:
            print(f"  Could not find both ducts for {cfg['name']}; skipping.")
            pair_results.append({"name": cfg["name"], "skipped": True})
            continue
        print(f"  Top duct:    y={top['y']}  L={top['left_x']} R={top['right_x']}")
        print(f"  Bottom duct: y={bottom['y']}  L={bottom['left_x']} R={bottom['right_x']}")

        corners, p1, p2 = trace_pair(
            binary, top, bottom,
            endpoint=cfg["endpoint"],
            direction=cfg["direction"],
            debug_canvas=debug_canvas,
        )
        draw_layer2_corners(overlay, corners)
        pair_results.append({
            "name":    cfg["name"],
            "skipped": False,
            "start":   list(corners[0]),
            "target":  list(corners[-1]),
            "corners": [list(c) for c in corners],
            "phase1":  {"reason": p1["reason"], "turn_dir": p1["turn_dir"]},
            "phase2":  None if p2 is None else {
                "reason":   p2["reason"],
                "turn_dir": p2["turn_dir"],
                "seed_left":  p2.get("seed_left"),
                "seed_right": p2.get("seed_right"),
            },
        })

    png_out  = output_dir / f"{pdf_in.stem}_step3_layer2.png"
    json_out = output_dir / f"{pdf_in.stem}_step3_layer2.json"
    cv2.imwrite(str(png_out), overlay)
    with open(json_out, "w") as f:
        json.dump({"pairs": pair_results}, f, indent=2)

    print(f"\n  PNG:  {png_out.relative_to(project_root)}")
    print(f"  JSON: {json_out.relative_to(project_root)}")
    if debug_canvas is not None:
        dbg_out = output_dir / f"{pdf_in.stem}_step3_layer2_debug.png"
        cv2.imwrite(str(dbg_out), debug_canvas)
        print(f"  DBG:  {dbg_out.relative_to(project_root)}")