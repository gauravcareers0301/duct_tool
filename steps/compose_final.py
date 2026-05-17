# steps/compose_final.py
"""
Step 6 (deliverable): Compose the final full-page PNG.

The annotated drawing (output/<stem>_step3_layer2.png) shows only the cropped
drawing region. For the final deliverable image we want the full PDF page --
including the title block, general notes, and plan notes -- with the duct
annotations overlaid in the correct location.

Approach:
  1. Rasterize page 1 of the input PDF at 300 DPI (NO crop).
  2. Load the annotated cropped image from Step 3 Layer 2.
  3. Paste the cropped image back onto the full page at (CROP_LEFT, CROP_TOP).
  4. Save as output/<stem>_final.png.

The cropped image already contains both the original drawing content AND the
blue duct annotation lines, so pasting it back simply replaces that region of
the full page with the annotated version. Everything outside the crop region
(title block, notes, grid border) is preserved untouched.
"""
import sys
from pathlib import Path

import cv2

# Reuse the loader and the canonical crop offsets from Step 1
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "steps"))
from load import load_pdf_page, find_input_pdf, CROP_LEFT, CROP_TOP


def compose_final(pdf_path, annotated_cropped_path, output_path, dpi=300):
    """Paste the annotated cropped image back onto the full PDF page."""
    # 1. Full-page raster (no crop)
    doc, page, full_image, zoom = load_pdf_page(pdf_path, dpi=dpi)
    doc.close()
    fh, fw = full_image.shape[:2]
    print(f"  Full page: {fw} x {fh}")

    # 2. Annotated cropped image
    annotated = cv2.imread(str(annotated_cropped_path))
    if annotated is None:
        raise FileNotFoundError(f"Could not read {annotated_cropped_path}")
    ah, aw = annotated.shape[:2]
    print(f"  Annotated cropped: {aw} x {ah}")
    print(f"  Pasting at offset: ({CROP_LEFT}, {CROP_TOP})")

    # 3. Sanity check the paste fits
    if CROP_LEFT + aw > fw or CROP_TOP + ah > fh:
        raise ValueError(
            f"Annotated region won't fit. Paste box would be "
            f"({CROP_LEFT}, {CROP_TOP}) -> ({CROP_LEFT + aw}, {CROP_TOP + ah}) "
            f"but page is {fw} x {fh}."
        )

    # 4. Composite (simple overwrite -- the cropped image already includes
    #    the original underlying drawing content).
    composed = full_image.copy()
    composed[CROP_TOP:CROP_TOP + ah, CROP_LEFT:CROP_LEFT + aw] = annotated

    # 5. Save
    cv2.imwrite(str(output_path), composed)
    print(f"  Saved: {output_path}")


if __name__ == "__main__":
    input_dir  = project_root / "input"
    output_dir = project_root / "output"
    output_dir.mkdir(exist_ok=True)

    pdf_in = find_input_pdf(input_dir)
    stem = pdf_in.stem

    annotated_path = output_dir / f"{stem}_step3_layer2.png"
    if not annotated_path.exists():
        print(f"ERROR: missing {annotated_path.name}. Run detect_layer_2.py first.")
        sys.exit(1)

    final_path = output_dir / f"{stem}_final.png"

    print(f"Composing final deliverable for {pdf_in.name}...")
    compose_final(pdf_in, annotated_path, final_path)
    print("Done.")