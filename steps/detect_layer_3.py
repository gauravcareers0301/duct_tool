# steps/detect_layer_3.py
"""
Step 3 -- Layer 3: Join the two LEFTMOST horizontal ducts into one connected
path. Mirror of Layer 2's dining-area logic, but walking RIGHTWARD from the
right endpoints of the two ducts, and the vertical section is on the right.

Reuses helpers from detect_layer_2 (binarize, walk_horizontal, walk_vertical,
draw functions).
"""
import json
import sys
from pathlib import Path

import cv2


# =============================================================================
# KNOBS for Layer 3
# =============================================================================
# These leftmost ducts have height ~44 px (vs dining's 82 px), so values are
# smaller than Layer 2's defaults. The shared helpers in detect_layer_2 use
# THEIR OWN INITIAL_JUMP_PX module-level value, which we override below.

# Override the shared layer-2 knob for this run.
# Set to ~1x duct height for these smaller ducts.
LAYER3_INITIAL_JUMP_PX = 70
# =============================================================================


def select_left_pair(ducts):
    """
    Pick the two right-stub fragments in the scullery area. These are the
    short Layer 1 segments closest to the vertical riser; their right
    endpoints sit at the entrance to the elbow turn -- the correct start
    points for tracing the U-shape.
    """
    candidates = []
    for d in ducts:
        # Stubs are short, far right of the area, in the same y band.
        if not (100 <= d["length_px"] <= 400):   continue
        if not (2400 <= d["left_x"]  <= 2700):   continue
        if not (2600 <= d["right_x"] <= 2900):   continue
        if not (1400 <= d["y"]       <= 2300):   continue
        candidates.append(d)
    candidates.sort(key=lambda d: d["y"])
    if len(candidates) < 2:
        return None, None
    return candidates[0], candidates[1]


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "steps"))
    from load import load_and_crop, find_input_pdf
    # Import shared logic from Layer 2
    import detect_layer_2 as L2

    # Override Layer 2's initial-jump knob for this layer's smaller ducts.
    L2.INITIAL_JUMP_PX = LAYER3_INITIAL_JUMP_PX

    output_dir = project_root / "output"
    pdf_in = find_input_pdf(project_root / "input")

    print(f"Loading & cropping {pdf_in.name}...")
    doc, page, image, zoom, crop_box = load_and_crop(pdf_in, dpi=300)
    doc.close()

    binary = L2.binarize(image)

    # ---- Load Layer 1 results ----
    layer1_path = output_dir / f"{pdf_in.stem}_step3_layer1.json"
    if not layer1_path.exists():
        print(f"  ERROR: Layer 1 output not found: {layer1_path}")
        sys.exit(1)
    with open(layer1_path) as f:
        layer1_ducts = json.load(f)
    print(f"  Loaded {len(layer1_ducts)} Layer 1 ducts")

    # ---- Load Layer 2 results to overlay on the same image ----
    layer2_path = output_dir / f"{pdf_in.stem}_step3_layer2.json"
    layer2_corners = []
    if layer2_path.exists():
        with open(layer2_path) as f:
            layer2_data = json.load(f)
        layer2_corners = [tuple(c) for c in layer2_data.get("corners", [])]
        print(f"  Loaded Layer 2: {len(layer2_corners)} corners")
    else:
        print(f"  WARN: Layer 2 output not found; will draw only Layer 3.")

    # ---- Pick the leftmost pair ----
    top, bottom = select_left_pair(layer1_ducts)
    if top is None or bottom is None:
        print("  Could not find both leftmost ducts; aborting.")
        for d in layer1_ducts:
            print(f"    candidate: y={d['y']} L={d['left_x']} R={d['right_x']} "
                  f"len={d['length_px']} h={d['height_px']}")
        sys.exit(1)
    print(f"  Top duct:    y={top['y']}  R=({top['right_x']}, {top['y']})  "
          f"walls top={top['top_y']} bot={top['bottom_y']}")
    print(f"  Bottom duct: y={bottom['y']}  R=({bottom['right_x']}, {bottom['y']})")

    # Jump size = full top duct's diameter
    jump_px = max(10, top["height_px"])
    print(f"  Jump size: {jump_px} px (top duct height {top['height_px']})")
    print(f"  Initial jump: {L2.INITIAL_JUMP_PX} px")

    # Start at the RIGHT endpoint, target the bottom duct's RIGHT endpoint.
    start_xy  = (top["right_x"],    top["y"])
    target_xy = (bottom["right_x"], bottom["y"])

    debug_canvas = image.copy() if L2.DEBUG_OVERLAY else None
    if debug_canvas is not None:
        L2.draw_layer1(debug_canvas, [top, bottom])
        cv2.circle(debug_canvas, target_xy, 14, (0, 0, 255), 3)

    # --- Phase 1: walk RIGHT (direction=+1) from top duct's right endpoint ---
    print("\nPhase 1: walking RIGHT from top duct's right endpoint...")
    p1 = L2.walk_horizontal(
        binary,
        start_x=top["right_x"], start_y=top["y"],
        jump_px=jump_px,
        seed_top_y=top["top_y"], seed_bottom_y=top["bottom_y"],
        direction=+1,
        debug_canvas=debug_canvas,
    )
    corner1 = (p1["corner_x"], p1["corner_y"])
    print(f"  reason: {p1['reason']}, turn_dir: {p1['turn_dir']}, "
          f"corner1={corner1}, samples={len(p1['samples'])}")
    print(f"  re-seeded walls after initial jump: "
          f"top={p1.get('seed_top_y')}, bottom={p1.get('seed_bottom_y')}")

    # --- Phase 2: walk vertically from corner1 ---
    if p1["turn_dir"] == 0:
        print("  Phase 1 did not detect an elbow; snapping straight to target.")
        corner2 = corner1
        p2 = None
    else:
        print(f"\nPhase 2: walking {'down' if p1['turn_dir']>0 else 'up'} from corner1...")
        p2 = L2.walk_vertical(
            binary, corner_x=corner1[0], corner_y=corner1[1],
            jump_px=jump_px, turn_dir=p1["turn_dir"],
            debug_canvas=debug_canvas,
        )
        corner2 = (p2["corner_x"], p2["corner_y"])
        print(f"  reason: {p2['reason']}, turn_dir: {p2['turn_dir']}, "
              f"corner2={corner2}, samples={len(p2['samples'])}")
        if p2.get("seed_left") is not None:
            print(f"  vertical-section walls: left={p2['seed_left']}, right={p2['seed_right']}")

    # --- Phase 3: corner2 -> target (horizontal back to bottom duct's right endpoint) ---
    corner2_snapped = (corner2[0], target_xy[1])
    print(f"\nPhase 3: corner2 {corner2} -> snapped {corner2_snapped} -> target {target_xy}")

    corners = [start_xy, corner1, corner2_snapped, target_xy]
    print(f"\nFinal corners: {corners}")

    # ---- Render the overlay: Layer 1 + Layer 2 + Layer 3 ----
    overlay = image.copy()
    L2.draw_layer1(overlay, layer1_ducts)
    if layer2_corners:
        L2.draw_layer2_corners(overlay, layer2_corners)
    L2.draw_layer2_corners(overlay, corners)

    if debug_canvas is not None:
        for c in corners:
            cv2.circle(debug_canvas, c, 8, L2.DBG_CORNER, 2)

    png_out  = output_dir / f"{pdf_in.stem}_step3_layer3.png"
    json_out = output_dir / f"{pdf_in.stem}_step3_layer3.json"
    cv2.imwrite(str(png_out), overlay)
    with open(json_out, "w") as f:
        json.dump({
            "start":   list(start_xy),
            "target":  list(target_xy),
            "corners": [list(c) for c in corners],
            "phase1": {
                "reason":   p1["reason"],
                "turn_dir": p1["turn_dir"],
            },
            "phase2": None if p2 is None else {
                "reason":   p2["reason"],
                "turn_dir": p2["turn_dir"],
                "seed_left":  p2.get("seed_left"),
                "seed_right": p2.get("seed_right"),
            },
        }, f, indent=2)

    print(f"\n  PNG:  {png_out.relative_to(project_root)}")
    print(f"  JSON: {json_out.relative_to(project_root)}")
    if debug_canvas is not None:
        dbg_out = output_dir / f"{pdf_in.stem}_step3_layer3_debug.png"
        cv2.imwrite(str(dbg_out), debug_canvas)
        print(f"  DBG:  {dbg_out.relative_to(project_root)}")