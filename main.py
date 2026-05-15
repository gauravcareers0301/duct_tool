"""
HVAC Duct Detection Tool - MVP

Phase 1:
- Read HVAC PDF
- Convert first page into image
- Load image using OpenCV
"""

import os
import fitz
import cv2


# -----------------------------
# Project Paths
# -----------------------------

INPUT_DIR = "input"
OUTPUT_DIR = "output"

PDF_FILE_NAME = "sample.pdf"

PDF_PATH = os.path.join(INPUT_DIR, PDF_FILE_NAME)

OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, "page1.png")


# -----------------------------
# PDF -> Image Conversion
# -----------------------------

def convert_pdf_to_image(pdf_path, output_image_path):

    # Open PDF
    doc = fitz.open(pdf_path)

    # Get first page
    page = doc[0]

    # High resolution rendering
    zoom_matrix = fitz.Matrix(3, 3)

    # Convert page into image
    pix = page.get_pixmap(matrix=zoom_matrix)

    # Save image
    pix.save(output_image_path)

    print(f"PDF converted successfully.")
    print(f"Image saved at: {output_image_path}")


# -----------------------------
# Load Image using OpenCV
# -----------------------------

def load_image(image_path):

    # Read image
    image = cv2.imread(image_path)

    # Check if image loaded correctly
    if image is None:
        print("ERROR: Failed to load image.")
        return None

    print("Image loaded successfully.")

    # Print image dimensions
    print(f"Image Shape: {image.shape}")

    return image


# -----------------------------
# Image Preprocessing
# -----------------------------

def preprocess_image(image):
    """
    Prepares the HVAC drawing for duct/line detection.

    Steps:
    1. Convert color image to grayscale
    2. Apply thresholding to separate black drawing lines from white background
    3. Save the processed image for debugging
    """

    # Convert BGR color image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Convert image into black/white binary image
    # THRESH_BINARY_INV makes black drawing lines become white
    _, binary = cv2.threshold(
        gray,
        200,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Save debug image
    debug_path = os.path.join(OUTPUT_DIR, "debug_binary.png")
    cv2.imwrite(debug_path, binary)

    print(f"Preprocessed binary image saved at: {debug_path}")

    return binary

# -----------------------------
# Detect and Highlight Ducts
# -----------------------------

def detect_ducts(binary_image, original_image):
    """
    Detects duct-like long line structures and highlights them in blue.

    Simple MVP approach:
    1. Focus only on floor plan area
    2. Detect horizontal and vertical duct-like lines
    3. Dilate lines to make them visually thicker
    4. Overlay blue highlight on original drawing
    """

    image_height, image_width = binary_image.shape

    # -----------------------------
    # Step 1: Focus on HVAC floor plan area
    # -----------------------------
    roi_x1 = int(image_width * 0.10)
    roi_y1 = int(image_height * 0.20)
    roi_x2 = int(image_width * 0.70)
    roi_y2 = int(image_height * 0.55)

    roi = binary_image[roi_y1:roi_y2, roi_x1:roi_x2]

    # Copy original ROI for annotation
    annotated_roi = original_image[roi_y1:roi_y2, roi_x1:roi_x2]

    # -----------------------------
    # Step 2: Extract horizontal duct-like lines
    # -----------------------------
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (60, 3)
    )

    horizontal_lines = cv2.morphologyEx(
        roi,
        cv2.MORPH_OPEN,
        horizontal_kernel,
        iterations=1
    )

    # -----------------------------
    # Step 3: Extract vertical duct-like lines
    # -----------------------------
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 60)
    )

    vertical_lines = cv2.morphologyEx(
        roi,
        cv2.MORPH_OPEN,
        vertical_kernel,
        iterations=1
    )

    # -----------------------------
    # Step 4: Combine horizontal and vertical lines
    # -----------------------------
    duct_mask = cv2.add(horizontal_lines, vertical_lines)

    # Make detected ducts thicker for visible highlighting
    highlight_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (9, 9)
    )

    duct_mask = cv2.dilate(
        duct_mask,
        highlight_kernel,
        iterations=1
    )

    # Save debug mask
    debug_path = os.path.join(OUTPUT_DIR, "debug_duct_mask.png")
    cv2.imwrite(debug_path, duct_mask)
    print(f"Duct mask saved at: {debug_path}")

    # -----------------------------
    # Step 5: Overlay blue highlight
    # -----------------------------
    blue_overlay = annotated_roi.copy()

    # Wherever duct_mask is white, color it blue
    blue_overlay[duct_mask > 0] = (255, 0, 0)

    # Blend original image with blue overlay
    blended = cv2.addWeighted(
        annotated_roi,
        0.70,
        blue_overlay,
        0.30,
        0
    )

    # Put blended ROI back into original image
    original_image[roi_y1:roi_y2, roi_x1:roi_x2] = blended

    # -----------------------------
    # Step 6: Save final output
    # -----------------------------
    output_path = os.path.join(
        OUTPUT_DIR,
        "duct_detection_debug.png"
    )

    cv2.imwrite(output_path, original_image)

    print(f"Duct highlighted output saved at: {output_path}")
# -----------------------------
# Main Execution
# -----------------------------

if __name__ == "__main__":

    # Create output folder if missing
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Check input PDF exists
    if not os.path.exists(PDF_PATH):

        print(f"ERROR: PDF not found at {PDF_PATH}")

    else:

        # Step 1 -> Convert PDF to image
        convert_pdf_to_image(PDF_PATH, OUTPUT_IMAGE_PATH)

        # Step 2 -> Load image using OpenCV
        image = load_image(OUTPUT_IMAGE_PATH)

        # Step 3 -> Preprocess image
        if image is not None:
            binary_image = preprocess_image(image)

        # Step 4 -> Detect duct-like regions
        detect_ducts(binary_image, image)
    