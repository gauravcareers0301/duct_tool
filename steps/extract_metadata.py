# steps/extract_metadata.py
"""
Step 4: Metadata Extraction.

For every detected duct (Layer 1 standalone or Layer 2 U-pair), produce:
  - id            : stable identifier (duct_1, duct_2, ...)
  - dimension     : "NN\"\u03c6" string (e.g. "14\"\u03c6")
  - pressure_class: Low | Medium | High
  - bboxes        : list of clickable [x0, y0, x1, y1] regions

Approach:
  1. Read Layer 1 results.
  2. For each Layer 1 anchor, do a directed OCR on a small crop just LEFT of
     the \u03c6 symbol to read the dimension digits. Fall back to Step 2 labels
     if the directed OCR fails.
  3. Merge Layer 2 U-pairs: each pair becomes one logical duct with 3 segments
     and 3 bboxes; its two source Layer 1 ducts are no longer standalone.
  4. Apply the pressure-class heuristic.
  5. Save sample_step4_ducts.json + sample_step4_ducts.csv.
"""
import csv
import json
import re
import sys
from pathlib import Path

import cv2
import pytesseract


# =============================================================================
# KNOBS
# =============================================================================
# Directed OCR crop just LEFT of each Layer 1 anchor (in pixels).
OCR_CROP_WIDTH  = 80     # how far left of anchor to crop
OCR_CROP_HEIGHT = 60     # vertical extent (centered on anchor_y)
OCR_CROP_PAD_RIGHT = 8   # stop this many px short of anchor_x (skip the symbol)

# Step 2 label fallback: max distance from anchor to consider a label
LABEL_FALLBACK_MAX_PX = 80

# Pressure-class heuristic
PRESSURE_RULES = [
    # (min_inches, max_inches, class)
    (0,  8,  "Low"),
    (9,  14, "Medium"),
    (15, 99, "High"),
]

# Half-thickness around centerlines for clickable bboxes
BBOX_HALF_HORIZ_PX = 35   # vertical thickness of horizontal segment hit-area
BBOX_HALF_VERT_PX  = 35   # horizontal thickness of vertical segment hit-area

# Room-name keyword -> human-readable area. We match the visible name text
# (KITCHEN, DINING, SCULLERY...) on the drawing, NOT the room number.
# Numbers like "104" are sometimes embedded in callouts/schedules at the
# wrong location; the visible name text is always written inside the room.
ROOM_KEYWORDS = {
    "KITCHEN":  "Kitchen",
    "DINING":   "Dining",
    "SCULLERY": "Scullery",
    "RESTROOM": "Restroom",
    "SERVICE":  "Service",
    "WALK-IN COOLER":  "Walk-in Cooler",
    "WALK-IN FREEZER": "Walk-in Freezer",
}

# Only use OCR-source room-name labels. PDF-source ones can come from
# rotated text in schedules/legends and sit at misleading coordinates.
ROOM_LABEL_SOURCE = "ocr"

# Max distance (in px) to consider a room-name label as the duct's room.
AREA_MATCH_MAX_PX = 1500

# Scale factor: 300 DPI raster, drawing scale 1/4" = 1'-0", so 1 ft = 75 px.
SCALE_PX_PER_FT = 75.0
# =============================================================================


# Match the digits at the end of an OCR string before an optional " or \u03c6 marker.
RE_DIM_DIGITS = re.compile(r'(\d{1,2})\s*[\"\'\u201d]?\s*[\u03c6\u00f8\u00d8\u00a2@\u03d5\u03a6]?')


def crop_dimension_region(image, anchor_x, anchor_y):
    """Return a small grayscale crop just LEFT of (anchor_x, anchor_y)."""
    h, w = image.shape[:2]
    x1 = max(0, anchor_x - OCR_CROP_PAD_RIGHT)
    x0 = max(0, x1 - OCR_CROP_WIDTH)
    y0 = max(0, anchor_y - OCR_CROP_HEIGHT // 2)
    y1 = min(h, anchor_y + OCR_CROP_HEIGHT // 2)
    crop = image[y0:y1, x0:x1]
    return crop


def ocr_dimension_digits(crop):
    """OCR a small crop with digit whitelist; return the int (or None)."""
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    # Slight upscale + threshold helps Tesseract on tiny digits
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cfg = '--psm 7 -c tessedit_char_whitelist=0123456789'
    text = pytesseract.image_to_string(binary, config=cfg).strip()
    if not text:
        return None
    m = re.search(r'\d{1,2}', text)
    if not m:
        return None
    try:
        n = int(m.group(0))
        if 3 <= n <= 48:
            return n
    except ValueError:
        pass
    return None


def nearest_label_diameter(labels, anchor_x, anchor_y, max_px=LABEL_FALLBACK_MAX_PX):
    """Fall back: find the closest duct_round label and parse its digits."""
    best = None
    best_dist = max_px
    for l in labels:
        if l.get("kind") != "duct_round":
            continue
        lx, ly = l["center_px"]
        d = ((lx - anchor_x) ** 2 + (ly - anchor_y) ** 2) ** 0.5
        if d < best_dist:
            best = l
            best_dist = d
    if best is None:
        return None
    m = RE_DIM_DIGITS.search(best["text"])
    if not m:
        return None
    try:
        n = int(m.group(1))
        if 3 <= n <= 48:
            return n
    except (ValueError, TypeError):
        return None


def pressure_class(diameter_inches):
    if diameter_inches is None:
        return None
    for lo, hi, cls in PRESSURE_RULES:
        if lo <= diameter_inches <= hi:
            return cls
    return None


def fmt_dimension(diameter_inches):
    if diameter_inches is None:
        return None
    return f'{diameter_inches}"\u03c6'


# ----- bbox helpers --------------------------------------------------------

def bbox_horizontal(x0, x1, y, half=BBOX_HALF_HORIZ_PX):
    lo, hi = sorted((x0, x1))
    return [lo, y - half, hi, y + half]


def bbox_vertical(x, y0, y1, half=BBOX_HALF_VERT_PX):
    lo, hi = sorted((y0, y1))
    return [x - half, lo, x + half, hi]


def centroid_of_bboxes(bboxes):
    """Average center of all bboxes -- a reasonable 'where is this duct' point."""
    if not bboxes:
        return None
    cxs = [(b[0] + b[2]) / 2 for b in bboxes]
    cys = [(b[1] + b[3]) / 2 for b in bboxes]
    return (sum(cxs) / len(cxs), sum(cys) / len(cys))


def dist_point_to_bbox(cx, cy, bbox):
    """Minimum Euclidean distance from point (cx,cy) to a bbox [x0,y0,x1,y1].
    Returns 0 if the point is inside the bbox."""
    x0, y0, x1, y1 = bbox
    dx = max(x0 - cx, 0, cx - x1)
    dy = max(y0 - cy, 0, cy - y1)
    return (dx * dx + dy * dy) ** 0.5


def nearest_room_name(centroid, labels, max_px=AREA_MATCH_MAX_PX):
    """
    Return the room-name area whose visible text bbox is nearest to the duct's
    centroid (distance = 0 if centroid is inside the room-name's text bbox).

    Matches OCR-source labels whose text equals one of ROOM_KEYWORDS' keys
    (case-insensitive). PDF-source room names are ignored because they often
    come from rotated text in schedules/legends at misleading coordinates.
    """
    if centroid is None or not labels:
        return None
    cx, cy = centroid
    best_name = None
    best_dist = max_px
    for l in labels:
        if l.get("source") != ROOM_LABEL_SOURCE:
            continue
        txt = l.get("text", "").strip().upper()
        if txt not in ROOM_KEYWORDS:
            continue
        d = dist_point_to_bbox(cx, cy, l["bbox_px"])
        if d < best_dist:
            best_name = ROOM_KEYWORDS[txt]
            best_dist = d
    return best_name


def position_from_centroid(centroid, bounds):
    """
    Return (position_x, position_y) using ADAPTIVE bounds (the min/max x,y
    of all duct centroids), so positions are relative to where ducts actually
    live in the drawing -- not to the full image (which has empty margins).

    bounds = (x_min, x_max, y_min, y_max).
    position_x is one of: 'left', 'center', 'right'.
    position_y is one of: 'top', 'middle', 'bottom'.
    """
    if centroid is None or bounds is None:
        return None, None
    cx, cy = centroid
    x_min, x_max, y_min, y_max = bounds
    x_span = max(1, x_max - x_min)
    y_span = max(1, y_max - y_min)
    fx = (cx - x_min) / x_span
    fy = (cy - y_min) / y_span
    px = "left"   if fx < 1/3 else "right"  if fx > 2/3 else "center"
    py = "top"    if fy < 1/3 else "bottom" if fy > 2/3 else "middle"
    return px, py


def length_ft(length_px):
    """Convert pixel length to feet using the drawing scale, rounded to 1 dp."""
    if length_px is None or length_px <= 0:
        return None
    return round(length_px / SCALE_PX_PER_FT)


# ----- main ----------------------------------------------------------------

def find_layer1_match(layer1, x, y, tol=5):
    """Return index of a Layer 1 duct whose anchor or endpoint matches (x, y)."""
    for i, d in enumerate(layer1):
        if abs(d["left_x"] - x) <= tol and abs(d["y"] - y) <= tol:
            return i
        if abs(d["right_x"] - x) <= tol and abs(d["y"] - y) <= tol:
            return i
    return None


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "steps"))
    from load import load_and_crop, find_input_pdf

    input_dir  = project_root / "input"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    pdf_in = find_input_pdf(input_dir)
    print(f"Loading & cropping {pdf_in.name}...")
    doc, page, image, zoom, crop_box = load_and_crop(pdf_in, dpi=300)
    doc.close()

    stem = pdf_in.stem
    layer1_path = output_dir / f"{stem}_step3_layer1.json"
    layer2_path = output_dir / f"{stem}_step3_layer2.json"
    labels_path = output_dir / f"{stem}_step2_labels.json"

    if not layer1_path.exists():
        print(f"  ERROR: missing {layer1_path.name}; run detect_layer_1 first.")
        sys.exit(1)
    with open(layer1_path) as f:
        layer1 = json.load(f)
    print(f"  Loaded Layer 1: {len(layer1)} ducts")

    layer2_pairs = []
    if layer2_path.exists():
        with open(layer2_path) as f:
            l2 = json.load(f)
        layer2_pairs = [p for p in l2.get("pairs", []) if not p.get("skipped")]
        print(f"  Loaded Layer 2: {len(layer2_pairs)} U-pairs")
    else:
        print("  WARN: no Layer 2 output; standalone ducts only.")

    labels = []
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)
        print(f"  Loaded Step 2 labels: {len(labels)}")
    else:
        print("  WARN: no Step 2 labels; directed-OCR only for dimensions.")

    # ---- Phase A: dimension per Layer 1 duct via directed OCR ----
    print("\nResolving dimensions via directed OCR...")
    layer1_diam = [None] * len(layer1)
    for i, d in enumerate(layer1):
        ax, ay = d["anchor_x"], d["anchor_y"]
        crop = crop_dimension_region(image, ax, ay)
        n = ocr_dimension_digits(crop)
        src = "ocr"
        if n is None:
            n = nearest_label_diameter(labels, ax, ay)
            src = "label" if n is not None else "none"
        layer1_diam[i] = n
        print(f"  duct anchor=({ax:5d},{ay:5d})  diameter={n}  via={src}")

    # ---- Phase B: figure out which Layer 1 indices each U-pair consumes ----
    pair_members = {}
    for p in layer2_pairs:
        sx, sy = p["start"]
        tx, ty = p["target"]
        top_idx    = find_layer1_match(layer1, sx, sy)
        bottom_idx = find_layer1_match(layer1, tx, ty)
        pair_members[p["name"]] = (top_idx, bottom_idx)
        print(f"  pair {p['name']}: top_idx={top_idx}, bottom_idx={bottom_idx}")

    consumed = set()
    for top_idx, bottom_idx in pair_members.values():
        if top_idx is not None:    consumed.add(top_idx)
        if bottom_idx is not None: consumed.add(bottom_idx)

    # ---- Phase C: assemble final ducts list ----
    ducts = []
    next_id = 1

    # Standalone Layer 1 ducts (not part of a U-pair)
    for i, d in enumerate(layer1):
        if i in consumed:
            continue
        diam = layer1_diam[i]
        bbox = bbox_horizontal(d["left_x"], d["right_x"], d["y"])
        bboxes = [bbox]
        centroid = centroid_of_bboxes(bboxes)
        centerline_px = abs(d["right_x"] - d["left_x"])
        ducts.append({
            "id": f"duct_{next_id}",
            "orientation":    "horizontal",
            "dimension":      fmt_dimension(diam),
            "pressure_class": pressure_class(diam),
            "area":           nearest_room_name(centroid, labels),
            "length_ft":      length_ft(centerline_px),
            "bboxes":         bboxes,
            "_centroid":      centroid,
        })
        next_id += 1

    # U-pair ducts -- split into 3 SEPARATE entries each (top h, vertical, bottom h).
    # The vertical connector has no labeled dimension in plan view, but it carries
    # the same airflow as the horizontals so it inherits their pressure class AND area.
    for p in layer2_pairs:
        top_idx, bottom_idx = pair_members[p["name"]]
        diam_top    = layer1_diam[top_idx]    if top_idx is not None    else None
        diam_bottom = layer1_diam[bottom_idx] if bottom_idx is not None else None

        # Inherited pressure for the vertical: from whichever horizontal we know.
        inherited_diam = diam_top or diam_bottom
        if diam_top and diam_bottom:
            inherited_diam = max(diam_top, diam_bottom)
        inherited_pressure = pressure_class(inherited_diam)

        corners = p["corners"]   # [start, c1, c2_snapped, target]
        (sx, sy), (c1x, c1y), (c2x, c2y), (tx, ty) = corners

        # Extents of each segment (covering the full visible blue line)
        if top_idx is not None:
            top_left  = min(layer1[top_idx]["left_x"],  layer1[top_idx]["right_x"], c1x)
            top_right = max(layer1[top_idx]["left_x"],  layer1[top_idx]["right_x"], c1x)
            top_centerline_px = abs(layer1[top_idx]["right_x"] - layer1[top_idx]["left_x"])
        else:
            top_left, top_right = sorted((sx, c1x))
            top_centerline_px = abs(top_right - top_left)

        if bottom_idx is not None:
            bot_left  = min(layer1[bottom_idx]["left_x"], layer1[bottom_idx]["right_x"], c2x)
            bot_right = max(layer1[bottom_idx]["left_x"], layer1[bottom_idx]["right_x"], c2x)
            bot_centerline_px = abs(layer1[bottom_idx]["right_x"] - layer1[bottom_idx]["left_x"])
        else:
            bot_left, bot_right = sorted((c2x, tx))
            bot_centerline_px = abs(bot_right - bot_left)

        vertical_centerline_px = abs(c2y - c1y)

        # --- Entry: top horizontal ---
        top_bbox = bbox_horizontal(top_left, top_right, sy)
        top_centroid = centroid_of_bboxes([top_bbox])
        top_area = nearest_room_name(top_centroid, labels)

        # --- Entry: bottom horizontal ---
        bot_bbox = bbox_horizontal(bot_left, bot_right, ty)
        bot_centroid = centroid_of_bboxes([bot_bbox])
        bot_area = nearest_room_name(bot_centroid, labels)

        # --- Entry: vertical connector ---
        vert_bbox = bbox_vertical(c1x, c1y, c2y)
        vert_centroid = centroid_of_bboxes([vert_bbox])
        vert_area = nearest_room_name(vert_centroid, labels)

        ducts.append({
            "id": f"duct_{next_id}",
            "orientation":    "horizontal",
            "dimension":      fmt_dimension(diam_top),
            "pressure_class": pressure_class(diam_top),
            "area":           top_area,
            "length_ft":      length_ft(top_centerline_px),
            "bboxes":         [top_bbox],
            "_centroid":      top_centroid,
        })
        next_id += 1
        ducts.append({
            "id": f"duct_{next_id}",
            "orientation":    "vertical",
            "dimension":      None,
            "pressure_class": inherited_pressure,
            "area":           vert_area,
            "length_ft":      length_ft(vertical_centerline_px),
            "bboxes":         [vert_bbox],
            "_centroid":      vert_centroid,
        })
        next_id += 1
        ducts.append({
            "id": f"duct_{next_id}",
            "orientation":    "horizontal",
            "dimension":      fmt_dimension(diam_bottom),
            "pressure_class": pressure_class(diam_bottom),
            "area":           bot_area,
            "length_ft":      length_ft(bot_centerline_px),
            "bboxes":         [bot_bbox],
            "_centroid":      bot_centroid,
        })
        next_id += 1

    # ---- Phase D: adaptive position bands from the duct centroids' bbox ----
    centroids = [d["_centroid"] for d in ducts if d.get("_centroid") is not None]
    if centroids:
        xs = [c[0] for c in centroids]
        ys = [c[1] for c in centroids]
        bounds = (min(xs), max(xs), min(ys), max(ys))
        print(f"\nPosition bounds (from duct centroids): "
              f"x={bounds[0]:.0f}..{bounds[1]:.0f}  y={bounds[2]:.0f}..{bounds[3]:.0f}")
    else:
        bounds = None

    for d in ducts:
        px, py = position_from_centroid(d.get("_centroid"), bounds)
        d["position_x"] = px
        d["position_y"] = py
        d.pop("_centroid", None)   # strip the internal helper

    # ---- Save JSON ----
    out_json = output_dir / f"{stem}_step4_ducts.json"
    with open(out_json, "w") as f:
        json.dump({"ducts": ducts}, f, indent=2)
    print(f"\n  JSON: {out_json.relative_to(project_root)}  ({len(ducts)} ducts)")

    # ---- Save CSV (flat columns for spreadsheet review) ----
    out_csv = output_dir / f"{stem}_step4_ducts.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "orientation", "dimension", "pressure_class",
                    "area", "position_x", "position_y", "length_ft"])
        for d in ducts:
            w.writerow([
                d["id"],
                d.get("orientation") or "",
                d["dimension"] or "",
                d["pressure_class"] or "",
                d.get("area") or "",
                d.get("position_x") or "",
                d.get("position_y") or "",
                d.get("length_ft") if d.get("length_ft") is not None else "",
            ])
    print(f"  CSV:  {out_csv.relative_to(project_root)}")

    # Console summary
    print("\nFinal ducts:")
    print(f"  {'id':10s}  {'orient':10s}  {'dim':8s}  {'pressure':8s}  "
          f"{'area':12s}  {'pos_x':7s}  {'pos_y':7s}  {'len_ft':>7s}")
    for d in ducts:
        print(f"  {d['id']:10s}  {str(d.get('orientation')):10s}  "
              f"{str(d['dimension']):8s}  {str(d['pressure_class']):8s}  "
              f"{str(d.get('area')):12s}  {str(d.get('position_x')):7s}  "
              f"{str(d.get('position_y')):7s}  {str(d.get('length_ft')):>7s}")