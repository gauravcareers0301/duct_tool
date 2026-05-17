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
# Allow the app to live in either project root or steps/ subdirectory.
if not (PROJECT_ROOT / "output").exists() and (PROJECT_ROOT.parent / "output").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
OUTPUT_DIR = PROJECT_ROOT / "output"

DEFAULT_STEM = "sample"
ANNOTATED_PNG = OUTPUT_DIR / f"{DEFAULT_STEM}_step3_layer2.png"
DUCTS_JSON    = OUTPUT_DIR / f"{DEFAULT_STEM}_step4_ducts.json"

# Browser render width. Smaller = faster transfer. 1100 is a good balance.
DISPLAY_WIDTH = 1100

# Click marker (drawn on the SMALL image, so coords are display-scale)
MARKER_RADIUS_DISPLAY = 6
MARKER_COLOR          = (255, 0, 0)

HIGHLIGHT_COLOR = (255, 0, 255)   # magenta
HIGHLIGHT_WIDTH = 4               # in display px (will look ~12 native px)
# =============================================================================


# ---------- Cached loaders -------------------------------------------------

@st.cache_data
def load_ducts(path_str: str):
    p = Path(path_str)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f).get("ducts", [])


@st.cache_data
def load_annotated_small(path_str: str, width: int):
    """Load + downscale ONCE. Cached so we never re-resize the big image."""
    p = Path(path_str)
    if not p.exists():
        return None, None, None
    big = Image.open(p).convert("RGB")
    nw, nh = big.size
    ratio = width / nw
    small = big.resize((width, int(nh * ratio)), Image.LANCZOS)
    return small, nw, nh


# ---------- Hit testing (always in NATIVE coords) --------------------------

def hit_test(click_x_native, click_y_native, ducts):
    for d in ducts:
        for x0, y0, x1, y1 in d["bboxes"]:
            if x0 <= click_x_native <= x1 and y0 <= click_y_native <= y1:
                return d
    return None


# ---------- Overlay drawing (on the SMALL image) ---------------------------

def overlay_on_small(small_img, native_w, click_native_xy, hit_duct):
    """Draw marker + highlight on the small image. Bboxes are scaled down."""
    img = small_img.copy()
    draw = ImageDraw.Draw(img)
    s = small_img.size[0] / native_w   # scale factor: native -> display

    if hit_duct is not None:
        for x0, y0, x1, y1 in hit_duct["bboxes"]:
            draw.rectangle(
                [x0 * s, y0 * s, x1 * s, y1 * s],
                outline=HIGHLIGHT_COLOR,
                width=HIGHLIGHT_WIDTH,
            )

    if click_native_xy is not None:
        cx, cy = click_native_xy
        dx, dy = cx * s, cy * s
        r = MARKER_RADIUS_DISPLAY
        draw.ellipse(
            [dx - r, dy - r, dx + r, dy + r],
            fill=MARKER_COLOR, outline=MARKER_COLOR,
        )

    return img


# ---------- App ------------------------------------------------------------

st.set_page_config(page_title="HVAC Duct Detection", layout="wide")
st.title("HVAC Duct Detection & Annotation")
st.caption("Click on any annotated duct in the drawing to see its details.")

ducts = load_ducts(str(DUCTS_JSON))
small_img, native_w, native_h = load_annotated_small(str(ANNOTATED_PNG), DISPLAY_WIDTH)

if ducts is None:
    st.error(f"Missing {DUCTS_JSON.relative_to(PROJECT_ROOT)}. Run the pipeline first.")
    st.stop()
if small_img is None:
    st.error(f"Missing {ANNOTATED_PNG.relative_to(PROJECT_ROOT)}. Run Step 3 first.")
    st.stop()

if "last_click_native" not in st.session_state:
    st.session_state.last_click_native = None
if "last_hit" not in st.session_state:
    st.session_state.last_hit = None

col_img, col_info = st.columns([3, 1])

with col_img:
    display_img = overlay_on_small(
        small_img, native_w,
        st.session_state.last_click_native,
        st.session_state.last_hit,
    )

    click = streamlit_image_coordinates(
        display_img,
        key="duct_click",
    )

    if click is not None:
        dx, dy = click["x"], click["y"]
        # display coords -> native coords (for hit testing)
        nx = int(dx * native_w / DISPLAY_WIDTH)
        ny = int(dy * native_w / DISPLAY_WIDTH)
        if st.session_state.last_click_native != (nx, ny):
            st.session_state.last_click_native = (nx, ny)
            st.session_state.last_hit = hit_test(nx, ny, ducts)
            st.rerun()

with col_info:
    st.subheader("Duct details")
    hit = st.session_state.last_hit
    if hit is None:
        if st.session_state.last_click_native is None:
            st.info("Click on a magenta-bordered duct in the drawing to see its details.")
        else:
            cx, cy = st.session_state.last_click_native
            st.warning(f"No duct at click ({cx}, {cy}). Try clicking directly on a blue line.")
    else:
        st.markdown(f"**ID**: `{hit['id']}`")
        if hit.get("orientation"):
            st.markdown(f"**Orientation**: {hit['orientation']}")
        st.markdown(f"**Dimension**: {hit.get('dimension') or '_Unknown_'}")
        st.markdown(f"**Pressure class**: {hit.get('pressure_class') or '_Unknown_'}")
        if hit.get("length_ft") is not None:
            st.markdown(f"**Length**: {hit['length_ft']} ft")
        if hit.get("area"):
            st.markdown(f"**Area**: {hit['area']}")
        px, py = hit.get("position_x"), hit.get("position_y")
        if px or py:
            st.markdown(f"**Position**: {py or '?'} / {px or '?'}")

    st.divider()
    st.caption(f"**{len(ducts)} ducts detected** in this drawing.")
    with st.expander("View all ducts"):
        for d in ducts:
            st.text(f"{d['id']}: {d.get('dimension') or '?'} / {d.get('pressure_class') or '?'}")