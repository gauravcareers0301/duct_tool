# steps/labels.py
"""
Step 2: Extract all labels from the cropped drawing region.
"""
import re
import json
import sys
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pytesseract
import fitz


# --- patterns ----------------------------------------------------------------

RE_ROUND_DUCT = re.compile(
    r'^\s*(\d{1,2})\s*'
    r'(?:'
    r'[”"\'`°]\s*[\dA-Za-z@¢°ɸØøφϕΦ]{0,2}'
    r'|'
    r'[¢@ɸØøφϕΦ]'
    r')'
    r'\s*$'
)
RE_RECT_DUCT  = re.compile(r'^\s*(\d{1,2})\s*[”"\']?\s*[xX]\s*(\d{1,2})\s*[”"\']?\s*$')
RE_INTEGER    = re.compile(r'^\s*(\d{1,4})\s*$')
RE_GRID_ALPHA = re.compile(r'^[A-Z]\.\d{1,2}$')
RE_HEIGHT_FULL = re.compile(r"""^\d{1,2}'\s*[-~]\s*\d{1,2}"$""")
RE_HEIGHT_FRAG = re.compile(r"""^(?:\d{1,2}'[-~]?|\d{1,2}")$""")
RE_ROOM_TAG   = re.compile(r'^[A-Z]\d{2}$')

DUCT_ROUND_INCHES = set(range(3, 49))


def _is_giant_grid_bubble(size):
    w, h = size
    return w >= 80 and h >= 80


def classify(text: str, bbox_size: tuple, source: str) -> str:
    t = text.strip()
    if not t:
        return "empty"
    w, h = bbox_size

    if RE_HEIGHT_FULL.match(t):
        return "height"
    if RE_HEIGHT_FRAG.match(t):
        return "other"
    if RE_RECT_DUCT.match(t):
        return "duct_rect"

    if source == "ocr":
        m = RE_ROUND_DUCT.match(t)
        if m:
            try:
                n = int(m.group(1))
                if n in DUCT_ROUND_INCHES:
                    return "duct_round"
            except (ValueError, TypeError):
                pass

    if RE_GRID_ALPHA.match(t):
        return "grid"
    if RE_INTEGER.match(t) and len(t.strip()) == 3 and _is_giant_grid_bubble((w, h)):
        return "grid"
    if RE_ROOM_TAG.match(t):
        return "room_tag"

    if RE_INTEGER.match(t):
        n = int(t)
        if 100 <= n <= 199:
            return "room_tag"
        if 1 <= n <= 25 and w < 60 and h < 60:
            return "note_num"
        if 50 <= n <= 2000 and w > h * 0.9:
            return "cfm"
        return "other"

    return "other"


# --- native PDF text ---------------------------------------------------------

def extract_native_text(page, zoom, crop_box):
    x0c, y0c, x1c, y1c = crop_box
    labels = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                txt = span.get("text", "").strip()
                if not txt:
                    continue
                px0, py0, px1, py1 = span["bbox"]
                fx0, fy0 = px0 * zoom, py0 * zoom
                fx1, fy1 = px1 * zoom, py1 * zoom
                cx, cy = (fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0
                if not (x0c <= cx <= x1c and y0c <= cy <= y1c):
                    continue
                bbox = (fx0 - x0c, fy0 - y0c, fx1 - x0c, fy1 - y0c)
                size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
                center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
                labels.append({
                    "text": txt,
                    "source": "pdf",
                    "conf": 100,
                    "bbox_px": bbox,
                    "center_px": center,
                    "size": size,
                    "kind": classify(txt, size, "pdf"),
                })
    return labels


# --- OCR ---------------------------------------------------------------------

def preprocess_for_ocr(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    return binary


def extract_ocr(image_cropped, min_conf=30):
    binary = preprocess_for_ocr(image_cropped)
    data = pytesseract.image_to_data(
        binary, config="--psm 11", output_type=pytesseract.Output.DICT
    )
    labels = []
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        if not txt:
            continue
        try:
            conf = int(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1
        if conf < min_conf:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        bbox = (x, y, x + w, y + h)
        size = (w, h)
        labels.append({
            "text": txt,
            "source": "ocr",
            "conf": conf,
            "bbox_px": bbox,
            "center_px": (x + w / 2.0, y + h / 2.0),
            "size": size,
            "kind": classify(txt, size, "ocr"),
        })
    return labels


# --- merge -------------------------------------------------------------------

def deduplicate(labels):
    pdf_labels = [l for l in labels if l["source"] == "pdf"]
    kept = list(pdf_labels)
    for ol in labels:
        if ol["source"] != "ocr":
            continue
        ox, oy = ol["center_px"]
        dup = False
        for pl in pdf_labels:
            px, py = pl["center_px"]
            if abs(ox - px) < 30 and abs(oy - py) < 30:
                dup = True
                break
        if not dup:
            kept.append(ol)
    return kept


# --- overlay -----------------------------------------------------------------

def draw_overlay(image_cropped, labels, out_path):
    overlay = image_cropped.copy()
    color_for = {
        "duct_round": (255, 0, 0),
        "duct_rect":  (0, 128, 255),
        "cfm":        (0, 200, 0),
        "note_num":   (180, 0, 180),
        "grid":       (255, 200, 0),
        "height":     (100, 100, 100),
        "room_tag":   (60, 60, 220),
        "other":      (220, 220, 220),
    }
    for l in labels:
        x0, y0, x1, y1 = (int(v) for v in l["bbox_px"])
        c = color_for.get(l["kind"], (200, 200, 200))
        thick = 5 if l["kind"].startswith("duct") else 3 if l["kind"] in ("cfm", "note_num") else 1
        cv2.rectangle(overlay, (x0, y0), (x1, y1), c, thick)
        if l["kind"].startswith("duct") or l["kind"] == "cfm":
            cv2.putText(overlay, l["text"], (x0, max(0, y0 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, c, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), overlay)


# --- main --------------------------------------------------------------------

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root / "steps"))
    from load import load_and_crop, find_input_pdf

    input_dir = project_root / "input"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    if len(sys.argv) > 1:
        pdf_in = Path(sys.argv[1])
        if not pdf_in.is_absolute():
            pdf_in = project_root / pdf_in
    else:
        pdf_in = find_input_pdf(input_dir)

    print(f"Loading & cropping {pdf_in.name}...")
    doc, page, image_cropped, zoom, crop_box = load_and_crop(pdf_in, dpi=300)
    print(f"  Cropped image: {image_cropped.shape[1]} x {image_cropped.shape[0]}")

    print("Extracting native PDF text...")
    pdf_labels = extract_native_text(page, zoom, crop_box)
    doc.close()
    print(f"  Native text labels (inside crop): {len(pdf_labels)}")

    print("Running OCR on cropped image (this takes ~15-40s)...")
    ocr_labels = extract_ocr(image_cropped, min_conf=30)
    print(f"  OCR tokens: {len(ocr_labels)}")

    all_labels = deduplicate(pdf_labels + ocr_labels)
    print(f"\nMerged labels (deduped): {len(all_labels)}")
    counts = Counter(l["kind"] for l in all_labels)
    for kind, n in counts.most_common():
        print(f"  {kind:12s} : {n}")

    for category in ("duct_round", "duct_rect", "cfm", "note_num", "room_tag"):
        cat = [l for l in all_labels if l["kind"] == category]
        if not cat:
            continue
        print(f"\n  {category} ({len(cat)}):")
        for l in cat[:40]:
            cx, cy = l["center_px"]
            print(f"    {l['text']!r:15s}  src={l['source']:3s} conf={l['conf']:3d}  at ({cx:.0f}, {cy:.0f})")
        if len(cat) > 40:
            print(f"    ... and {len(cat) - 40} more")

    # ---- Save outputs ----
    pdf_stem = pdf_in.stem
    json_out = output_dir / f"{pdf_stem}_step2_labels.json"
    overlay_out = output_dir / f"{pdf_stem}_step2_labels.png"

    with open(json_out, "w") as f:
        json.dump(all_labels, f, indent=2)
    print(f"\n  JSON: {json_out.relative_to(project_root)}")

    draw_overlay(image_cropped, all_labels, overlay_out)
    print(f"  PNG:  {overlay_out.relative_to(project_root)}")