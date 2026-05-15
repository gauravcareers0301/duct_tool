# steps/load.py
"""
Step 1: Load the PDF, rasterize page 1 at 300 DPI, and crop to the
drawing region (drops title block, notes columns, and margins).
"""
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
import cv2


# =============================================================================
# CROP BOX  --  edit these four numbers to change the cropped region.
# Coordinates are in pixels of the FULL 10800 x 7200 image.
# (Use output/crop_helper_grid.png as a reference to pick values.)
# =============================================================================
CROP_LEFT   = 300
CROP_TOP    = 700
CROP_RIGHT  = 9500
CROP_BOTTOM = 4500
# =============================================================================


def load_pdf_page(pdf_path, page_index=0, dpi=300):
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    page = doc[page_index]

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 4:
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    else:
        image_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return doc, page, image_bgr, zoom


def find_input_pdf(input_dir):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")
    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDF found in {input_dir}.")
    if len(pdfs) > 1:
        names = "\n  - ".join(p.name for p in pdfs)
        raise RuntimeError(f"Multiple PDFs in {input_dir}:\n  - {names}")
    return pdfs[0]


def crop_to_drawing(image_bgr):
    """Apply the CROP_* box defined at the top of this file."""
    h, w = image_bgr.shape[:2]
    x0 = max(0, min(CROP_LEFT, w))
    y0 = max(0, min(CROP_TOP, h))
    x1 = max(0, min(CROP_RIGHT, w))
    y1 = max(0, min(CROP_BOTTOM, h))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid crop box: ({x0},{y0}) -> ({x1},{y1})")
    return image_bgr[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)


def load_and_crop(pdf_path, dpi=300):
    """One-call entry point for later steps."""
    doc, page, image_full, zoom = load_pdf_page(pdf_path, dpi=dpi)
    image_cropped, crop_box = crop_to_drawing(image_full)
    return doc, page, image_cropped, zoom, crop_box


if __name__ == "__main__":
    import sys
    project_root = Path(__file__).resolve().parent.parent
    input_dir = project_root / "input"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    if len(sys.argv) > 1:
        pdf_in = Path(sys.argv[1])
        if not pdf_in.is_absolute():
            pdf_in = project_root / pdf_in
    else:
        pdf_in = find_input_pdf(input_dir)

    doc, page, image_full, zoom = load_pdf_page(pdf_in, dpi=300)
    image_cropped, crop_box = crop_to_drawing(image_full)
    x0, y0, x1, y1 = crop_box

    print(f"Loaded: {pdf_in.name}")
    print(f"  Full image:  {image_full.shape[1]} x {image_full.shape[0]}")
    print(f"  Crop box:    ({x0}, {y0}) -> ({x1}, {y1})")
    print(f"  Cropped:     {image_cropped.shape[1]} x {image_cropped.shape[0]}")

    cropped_out = output_dir / f"{pdf_in.stem}_step1_cropped.png"
    cv2.imwrite(str(cropped_out), image_cropped)
    print(f"  Saved: {cropped_out.relative_to(project_root)}")
    doc.close()