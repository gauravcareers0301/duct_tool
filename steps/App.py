# app.py
"""
HVAC Duct Detection & Annotation -- Interactive UI

Streamlit app that loads the pre-computed pipeline outputs and lets the user
click on the annotated drawing to see each duct's dimension and pressure class.

Run from the project root:
    streamlit run app.py
"""
import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates


# =============================================================================
# CONFIG
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR   = PROJECT_ROOT / "output"

DEFAULT_STEM = "sample"
ANNOTATED_PNG = OUTPUT_DIR / f"{DEFAULT_STEM}_step3_layer2.png"
DUCTS_JSON    = OUTPUT_DIR / f"{DEFAULT_STEM}_step4_ducts.json"

# How wide to render the image in the browser. Smaller = faster but less precise.
DISPLAY_WIDTH = 1100

# Click marker
MARKER_RADIUS_NATIVE = 12       # in original image pixels
MARKER_COLOR         = (255, 0, 0)  # red
HIGHLIGHT_COLOR      = (255, 255, 0)  # yellow outline around hit bbox
HIGHLIGHT_WIDTH      = 6
# =============================================================================


# ---------- Helpers --------------------------------------------------------

def load_ducts(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f).get("ducts", [])


def load_annotated_image(path: Path):
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")


def hit_test(click_x_native, click_y_native, ducts):
    """Return the first duct whose any bbox contains the click point."""
    for d in ducts:
        for x0, y0, x1, y1 in d["bboxes"]:
            if x0 <= click_x_native <= x1 and y0 <= click_y_native <= y1:
                return d
    return None


def overlay_marker_and_highlight(image_pil, click_native_xy, hit_duct):
    """Draw a click marker and optionally a yellow box around the hit duct's bbox."""
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)

    if hit_duct is not None:
        for x0, y0, x1, y1 in hit_duct["bboxes"]:
            draw.rectangle([x0, y0, x1, y1],
                           outline=HIGHLIGHT_COLOR, width=HIGHLIGHT_WIDTH)

    if click_native_xy is not None:
        cx, cy = click_native_xy
        r = MARKER_RADIUS_NATIVE
        draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                     fill=MARKER_COLOR, outline=MARKER_COLOR)

    return img


# ---------- App ------------------------------------------------------------

st.set_page_config(page_title="HVAC Duct Detection", layout="wide")

st.title("HVAC Duct Detection & Annotation")
st.caption("Click on any annotated duct in the drawing to see its details.")

# Load data
ducts = load_ducts(DUCTS_JSON)
annotated = load_annotated_image(ANNOTATED_PNG)

if ducts is None:
    st.error(f"Missing {DUCTS_JSON.relative_to(PROJECT_ROOT)}. "
             f"Run the pipeline first: "
             f"`python steps/load.py && python steps/labels.py && "
             f"python steps/detect_layer_1.py && python steps/detect_layer_2.py && "
             f"python steps/extract_metadata.py`")
    st.stop()

if annotated is None:
    st.error(f"Missing {ANNOTATED_PNG.relative_to(PROJECT_ROOT)}. "
             f"Run Step 3 (detect_layer_2.py) first.")
    st.stop()

native_w, native_h = annotated.size
scale = DISPLAY_WIDTH / native_w   # display_px = native_px * scale

# Persist last-click and last-hit across reruns
if "last_click_native" not in st.session_state:
    st.session_state.last_click_native = None
if "last_hit" not in st.session_state:
    st.session_state.last_hit = None

# Two-column layout: image left, details right
col_img, col_info = st.columns([3, 1])

with col_img:
    # Build the image with overlay (marker + highlight) BEFORE the click capture,
    # so the current rendered image shows the previous interaction's state.
    display_img = overlay_marker_and_highlight(
        annotated,
        st.session_state.last_click_native,
        st.session_state.last_hit,
    )

    # Capture click in DISPLAY coordinates
    click = streamlit_image_coordinates(
        display_img,
        width=DISPLAY_WIDTH,
        key="duct_click",
    )

    if click is not None:
        # Translate display click -> native image coords
        dx, dy = click["x"], click["y"]
        nx = int(dx / scale)
        ny = int(dy / scale)
        # Only update if the click point changed (avoids reprocessing on rerun)
        if st.session_state.last_click_native != (nx, ny):
            st.session_state.last_click_native = (nx, ny)
            st.session_state.last_hit = hit_test(nx, ny, ducts)
            st.rerun()

with col_info:
    st.subheader("Duct details")
    hit = st.session_state.last_hit
    if hit is None:
        if st.session_state.last_click_native is None:
            st.info("Click on a blue duct line in the drawing to see its details.")
        else:
            cx, cy = st.session_state.last_click_native
            st.warning(
                f"No duct at click ({cx}, {cy}). "
                f"Try clicking directly on a blue line."
            )
    else:
        st.markdown(f"**ID**: `{hit['id']}`")
        orient = hit.get("orientation")
        if orient:
            st.markdown(f"**Orientation**: {orient}")
        dim = hit.get("dimension")
        pcl = hit.get("pressure_class")
        area = hit.get("area")
        px = hit.get("position_x")
        py = hit.get("position_y")
        length_ft = hit.get("length_ft")
        st.markdown(f"**Dimension**: {dim if dim else '_Unknown_'}")
        st.markdown(f"**Pressure class**: {pcl if pcl else '_Unknown_'}")
        if length_ft is not None:
            st.markdown(f"**Length**: {length_ft} ft")
        if area:
            st.markdown(f"**Area**: {area}")
        if px or py:
            st.markdown(f"**Position**: {py or '?'} / {px or '?'}")

    st.divider()

    # Small summary
    st.caption(f"**{len(ducts)} ducts detected** in this drawing.")
    with st.expander("View all ducts"):
        for d in ducts:
            dim = d.get("dimension") or "?"
            pcl = d.get("pressure_class") or "?"
            st.text(f"{d['id']}: {dim} / {pcl}")