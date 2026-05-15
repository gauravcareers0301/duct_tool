# steps/detect_layer1.py
"""
Step 3 -- Layer 1: Detect ducts anchored by the slashed-zero diameter symbol.

For each detected anchor:
  - scan left and right to find the duct's horizontal extent
Saves:
  output/<stem>_step3_layer1.png   (visualization)
  output/<stem>_step3_layer1.json  (data for Layer 2)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# KNOBS
# =============================================================================
# Template-matching for ɸ symbol
TEMPLATE_CENTER     = (2942, 1204)
TEMPLATE_HALF_SIZE  = 18
MATCH_THRESHOLD     = 0.68
NMS_RADIUS_PX       = 25

# Anchor merging (template anchors + OCR labels)
ANCHOR_MERGE_PX     = 40

# Horizontal scan from anchor
EDGE_MIN_HEIGHT_PX         = 15
EDGE_MIN_WIDTH_PX          = 3
SCAN_MAX_PX                = 1000
SCAN_START_OFFSET_LEFT_PX  = 70
SCAN_START_OFFSET_RIGHT_PX = 20
# =============================================================================


def binarize(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    return binary


# ----- Layer 1: template matching ------------------------------------------

def extract_template(binary, center, half):
    cx, cy = center
    h, w = binary.shape
    x0 = max(0, cx - half); y0 = max(0, cy - half)
    x1 = min(w, cx + half); y1 = min(h, cy + half)
    return binary[y0:y1, x0:x1].copy()


def non_max_suppression(points, radius):
    pts = sorted(points, key=lambda p: -p[2])
    kept = []
    for x, y, s in pts:
        if any(abs(x - kx) < radius and abs(y - ky) < radius for kx, ky, _ in kept):
            continue
        kept.append((x, y, s))
    return kept


def find_template_anchors(binary):
    template = extract_template(binary, TEMPLATE_CENTER, TEMPLATE_HALF_SIZE)
    res = cv2.matchTemplate(binary, template, cv2.TM_CCOEFF_NORMED)
    th, tw = template.shape
    ys, xs = np.where(res >= MATCH_THRESHOLD)
    raw = [(int(x + tw // 2), int(y + th // 2), float(res[y, x]))
           for x, y in zip(xs, ys)]
    return non_max_suppression(raw, NMS_RADIUS_PX)


def merge_anchors(template_anchors, ocr_labels):
    anchors = []
    for x, y, s in template_anchors:
        anchors.append({"x": x, "y": y, "source": "template", "info": f"score={s:.2f}"})

    for l in ocr_labels:
        cx, cy = int(round(l["center_px"][0])), int(round(l["center_px"][1]))
        dup = any(abs(cx - a["x"]) < ANCHOR_MERGE_PX and
                  abs(cy - a["y"]) < ANCHOR_MERGE_PX for a in anchors)
        if not dup:
            anchors.append({"x": cx, "y": cy, "source": "ocr", "info": l["text"]})
    return anchors


# ----- Horizontal scan -----------------------------------------------------

def is_vertical_edge(binary, x, y_center,
                     min_height=EDGE_MIN_HEIGHT_PX,
                     min_width=EDGE_MIN_WIDTH_PX):
    h, w = binary.shape
    if not (0 <= x < w) or not (0 <= y_center < h):
        return False
    half_h = min_height // 2 + 2
    y0 = max(0, y_center - half_h)
    y1 = min(h, y_center + half_h)
    if x + min_width > w:
        return False
    for dx in range(min_width):
        col = binary[y0:y1, x + dx]
        run, best = 0, 0
        for v in col:
            if v > 0:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
        if best < min_height:
            return False
    return True


def scan_to_edge(binary, x_start, y, direction):
    w = binary.shape[1]
    x = x_start
    for _ in range(SCAN_MAX_PX):
        if not (0 <= x < w):
            break
        if is_vertical_edge(binary, x, y):
            return x
        x += direction
    return x


def find_duct_extent(binary, anchor):
    x, y = anchor["x"], anchor["y"]
    left_x  = scan_to_edge(binary, x - SCAN_START_OFFSET_LEFT_PX,  y, -1)
    right_x = scan_to_edge(binary, x + SCAN_START_OFFSET_RIGHT_PX, y, +1)
    return {
        "anchor_x":  x,
        "anchor_y":  y,
        "source":    anchor["source"],
        "info":      anchor["info"],
        "y":         y,
        "left_x":    left_x,
        "right_x":   right_x,
        "length_px": right_x - left_x,
    }


# ----- Vertical scan to find duct height -----------------------------------

def find_duct_height(binary, duct, sample_count=5):
    """
    Scan vertically up and down from the duct centerline at several x samples
    to find the top and bottom walls. Returns (top_y, bottom_y) -- the highest
    top and lowest bottom across samples (most generous bounds).
    """
    y = duct["y"]
    lx, rx = duct["left_x"], duct["right_x"]
    if rx <= lx:
        return y - 5, y + 5

    h, w = binary.shape
    sample_xs = np.linspace(lx + 20, rx - 20, sample_count).astype(int)
    sample_xs = [x for x in sample_xs if 0 <= x < w]

    MAX_SEARCH = 80  # don't look further than this for the wall
    tops, bottoms = [], []
    for sx in sample_xs:
        # scan up
        ty = y
        for _ in range(MAX_SEARCH):
            if ty <= 0:
                break
            if binary[ty, sx] > 0:
                tops.append(ty)
                break
            ty -= 1
        # scan down
        by = y
        for _ in range(MAX_SEARCH):
            if by >= h - 1:
                break
            if binary[by, sx] > 0:
                bottoms.append(by)
                break
            by += 1

    top_y    = int(np.median(tops))    if tops    else y - 5
    bottom_y = int(np.median(bottoms)) if bottoms else y + 5
    return top_y, bottom_y


# ----- Output rendering ----------------------------------------------------

def draw_results(image_bgr, ducts, out_path):
    overlay = image_bgr.copy()
    for d in ducts:
        y, lx, rx = d["y"], d["left_x"], d["right_x"]
        cv2.line(overlay, (lx, y), (rx, y), (255, 0, 0), 8)
        cv2.circle(overlay, (lx, y), 6, (255, 0, 0), -1)
        cv2.circle(overlay, (rx, y), 6, (255, 0, 0), -1)
    cv2.imwrite(str(out_path), overlay)


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

    # Anchor sources
    print("Layer 1: template matching for diameter symbol...")
    template_anchors = find_template_anchors(binary)
    print(f"  Template matches: {len(template_anchors)}")

    labels_path = output_dir / f"{pdf_in.stem}_step2_labels.json"
    duct_labels = []
    if labels_path.exists():
        with open(labels_path) as f:
            labels = json.load(f)
        duct_labels = [l for l in labels if l["kind"] == "duct_round"]
    print(f"  OCR duct_round labels: {len(duct_labels)}")

    anchors = merge_anchors(template_anchors, duct_labels)
    print(f"  Total anchors after merge: {len(anchors)}")

    # Find duct extent for each anchor
    print("\nDetecting horizontal duct extents...")
    ducts = [find_duct_extent(binary, a) for a in anchors]

    # Add height (top_y, bottom_y) for Layer 2 to use
    print("Measuring duct heights...")
    for d in ducts:
        top_y, bottom_y = find_duct_height(binary, d)
        d["top_y"]    = top_y
        d["bottom_y"] = bottom_y
        d["height_px"] = bottom_y - top_y

    print(f"\n  {'src':>8}  {'anchor':>18}  {'L':>5}  {'R':>5}  {'len':>5}  {'top':>5}  {'bot':>5}  {'ht':>4}")
    for d in ducts:
        print(f"  {d['source']:>8}  ({d['anchor_x']:5d},{d['anchor_y']:5d})  "
              f"{d['left_x']:>5}  {d['right_x']:>5}  {d['length_px']:>5}  "
              f"{d['top_y']:>5}  {d['bottom_y']:>5}  {d['height_px']:>4}")

    # Save outputs
    png_out = output_dir / f"{pdf_in.stem}_step3_layer1.png"
    json_out = output_dir / f"{pdf_in.stem}_step3_layer1.json"
    draw_results(image, ducts, png_out)
    with open(json_out, "w") as f:
        json.dump(ducts, f, indent=2)

    print(f"\n  PNG:  {png_out.relative_to(project_root)}")
    print(f"  JSON: {json_out.relative_to(project_root)}")