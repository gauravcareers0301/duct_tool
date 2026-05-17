# HVAC Duct Detection and Annotation

A pipeline that ingests an HVAC mechanical drawing (PDF), detects round
horizontal ducts and the U-shaped connections between them, and produces
an annotated image plus per-duct metadata (dimension, pressure class,
area, position, length). A Streamlit UI lets you click on any detected
duct to see its details.

Built as an interview deliverable. The focus was on a clean, end-to-end
working pipeline over chasing every edge case — limitations are
documented honestly in the [Known limitations](#known-limitations)
section.

---

## What's in the repo

```
duct_tool/
├── input/
│   └── sample.pdf                ← put your HVAC drawing here
├── output/                       ← all pipeline outputs land here
├── steps/
│   ├── load.py                   ← Step 1: PDF → 300 DPI cropped image
│   ├── labels.py                 ← Step 2: text label extraction
│   ├── detect_layer_1.py         ← Step 3a: round-duct detection
│   ├── detect_layer_2.py         ← Step 3b: U-shape tracing
│   ├── extract_metadata.py       ← Step 4: per-duct metadata
│   └── compose_final.py          ← Step 6: full-page deliverable
├── main.py                       ← runs steps 1-6 end-to-end
├── app.py                        ← Streamlit interactive UI
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- **Python 3.10+** (tested on 3.11)
- **Tesseract OCR** (used for label extraction):
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Windows: download installer from [tesseract-ocr/tesseract on GitHub](https://github.com/tesseract-ocr/tesseract)

### Install

```bash
git clone https://github.com/gauravcareers0301/duct_tool.git
cd duct_tool

python3 -m venv venv
source venv/bin/activate              # macOS/Linux
# .\venv\Scripts\activate              # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Run

### 1. Process the input drawing

Place an HVAC drawing PDF in `input/` (a sample is included), then:

```bash
python main.py
```

Runs all six pipeline steps in order. Takes ~30–60 seconds. Outputs
land in `output/`.

### 2. Launch the interactive UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Click on any annotated duct (blue
line) in the drawing to see its dimension, pressure class, area, and
length in the right-hand panel. A magenta box highlights the selected
duct.

---

## Output files

After `main.py` completes, `output/` contains:

| File | Purpose |
|------|---------|
| `sample_step1_cropped.png` | Cropped drawing region (title block + notes removed). |
| `sample_step2_labels.png` / `.json` | Extracted text labels (duct dimensions, CFM, room tags). |
| `sample_step3_layer1.png` / `.json` | Round ducts detected via diameter-symbol template matching. |
| `sample_step3_layer2.png` / `.json` | U-shaped connections traced between duct pairs. |
| `sample_step4_ducts.json` / `.csv` | Final per-duct metadata (used by the UI). |
| `sample_final.png` | **Final deliverable** — full PDF page with annotations, matching the original sheet layout. |

---

## Pipeline overview

```
input/sample.pdf
    │
    ▼
Step 1 — load & crop          (steps/load.py)
    │   Rasterizes the PDF at 300 DPI and crops to the drawing region.
    │
    ▼
Step 2 — labels                (steps/labels.py)
    │   Extracts text labels via PyMuPDF (native PDF text) + Tesseract
    │   OCR. Classifies labels into duct_round / duct_rect / cfm / etc.
    │
    ▼
Step 3 — duct detection
    │   Layer 1 (steps/detect_layer_1.py)
    │     • Template-matches the slashed-zero (ɸ) diameter symbol.
    │     • From each match, scans left/right for the duct's horizontal
    │       extent. Outputs blue centerlines through each round duct.
    │
    │   Layer 2 (steps/detect_layer_2.py)
    │     • Joins pairs of horizontal ducts into U-shaped paths through
    │       intervening fittings. Currently handles two pair locations:
    │       the dining-area loop and the scullery loop.
    │
    ▼
Step 4 — metadata extraction   (steps/extract_metadata.py)
    │   For each detected duct, resolves dimension (e.g. 14"ɸ),
    │   pressure class, area (Kitchen / Dining / Scullery / …),
    │   position bucket, centerline length in feet, orientation, and
    │   clickable bbox.
    │
    ▼
Step 6 — final composition     (steps/compose_final.py)
    │   Pastes the annotated drawing back onto the full PDF page,
    │   preserving the title block and notes for the final deliverable.
    │
    ▼
output/sample_final.png         (full-page annotated deliverable)
output/sample_step4_ducts.json  (per-duct metadata for the UI)
```

Step 5 is the Streamlit UI (`app.py`), launched separately.

---

## Design decisions

All values are tunable from the `KNOBS` block at the top of each step's
source file.

- **Round-duct dimensions** are read by directed OCR on a small crop
  immediately left of each ɸ symbol detected in Layer 1, with Step 2's
  general-page OCR as a fallback. Confidence is ~90% on the sample;
  misses are recorded as `null` (see Limitation #2 below).

- **Vertical connectors in U-shapes have no labeled ɸ dimension** in
  plan view (the symbol is drawn on the horizontal portions). Vertical
  entries therefore have `dimension: null`. Their pressure class is
  inherited from the horizontals of the same U-pair, since the air in
  all three segments belongs to the same physical run.

- **Area assignment** uses Euclidean distance from the duct centroid
  to the bounding box of the nearest *visible* room-name text
  (`KITCHEN`, `DINING`, `SCULLERY`, etc.). Only OCR-source room names
  are used — PDF-source room names can come from rotated text in
  schedules or legends at coordinates that don't reflect where the
  rooms actually are visually. A long duct crossing a room boundary
  may be assigned to whichever room's name text sits closer to its
  centroid.

- **Position** is reported as two independent buckets: `position_x`
  in {left, center, right} and `position_y` in {top, middle, bottom}.
  Bands are computed *adaptively* from the min/max x,y of all duct
  centroids (not the full image), so positions reflect where ducts
  actually live in the drawing.

- **Centerline length** is converted to feet using the drawing scale:
  300 DPI raster, drawing scale 1/4" = 1'-0", giving 75 px/ft. Values
  are rounded to whole feet.

- **Click hit-areas** are axis-aligned bounding boxes around each duct
  centerline (±35 px). Generous enough that clicks near a duct still
  register; hit-areas for nearby parallel ducts may overlap, in which
  case the first matching duct in the data is returned.

---

## Known limitations

The detection is tuned to the provided sample drawing. Four limitations
are worth calling out explicitly.

### 1. Rectangular and diagonal ducts are not detected

The pipeline targets **round horizontal ducts** only (those labeled
with the ɸ diameter symbol). It also handles the **vertical
connectors** between horizontal ducts in U-shape routings. What it
does NOT handle:

- **Rectangular ducts** labeled as `XX"×YY"` (a different labeling
  convention; would need a separate detection layer).
- **Diagonal (non-axis-aligned) ducts.** Both Layer 1's left/right
  horizontal scan and Layer 2's orthogonal wall-tracing assume
  axis-aligned ducts.

In the sample drawing, the `22"×14"` supply duct in the kitchen is
both rectangular AND runs diagonally at roughly 30–45° between grid
lines (referred to internally as `duct_6`). It is currently misread
as a short horizontal segment rather than the full diagonal run. A
correct detector for this case would anchor on the `XX"×YY"` label
via OCR, then search outward for two parallel dark lines at arbitrary
angle.

### 2. Dimension labels drawn outside the duct

Some ducts have their ɸ dimension symbol drawn **outside** the duct
line, with a leader pointing back to it (common in tight regions where
there isn't space to label the duct inline). The directed-OCR crop
window assumes the symbol sits on or immediately adjacent to the duct
centerline; when the label is offset to a clearer part of the drawing,
the crop misses it.

In the sample drawing this affects one duct near the bottom of the
kitchen area (a short connector, internally `duct_4`). It appears in
the output with `dimension: null` and is shown as "Unknown" in the UI.
It remains clickable and reports its geometry (orientation, area,
position, length) like any other duct.

A general fix would be to detect the leader lines and follow them back
to the associated label — out of scope for this submission.

### 3. Hard-coded anchor points and selection criteria

Several parameters in the detection pipeline are tied to coordinates
from the sample drawing:

- **Layer 1** uses a fixed template location to extract the ɸ-symbol
  reference image. The Step 1 crop region is also a fixed pixel box.
- **Layer 2** selects which duct pairs to trace by absolute pixel
  ranges (e.g. *"dining pair = ducts with `left_x` between 4800 and
  5100"*). Works for the sample; would need re-tuning for a new
  drawing.

A more general approach would derive these dynamically — for example,
detect the title block to locate the drawing region, and cluster
Layer 1 detections by endpoint coordinates to discover U-pair
candidates without absolute filters. Out of scope for this submission.

### 4. Pressure class is a heuristic, not a real classification

Real-world pressure class is defined by static pressure rating
(in. w.g.) and velocity per SMACNA standards — information that isn't
directly visible on this drawing and would normally come from the
equipment schedule or an adjacent sheet.

The current implementation uses duct diameter as a proxy:

- diameter ≤ 8" → "Low"
- diameter 9–14" → "Medium"
- diameter ≥ 15" → "High"

This is a deliberate simplification to produce a value the UI can
display. The intuition (smaller cross-section + same CFM → higher
velocity → higher pressure drop) is approximately right but should not
be relied on for any engineering purpose. A correct implementation
would read the pressure class from the project's mechanical schedule.

---

## Future enhancements (deferred by scope)

Things that would be worth doing in a longer-running version of this
project:

- **File watcher** — auto-trigger the pipeline when a new PDF is
  dropped into `input/`, with an `input_archive/` folder for
  processed files.
- **Dynamic anchor and crop discovery** — eliminate hard-coded pixel
  ranges so the pipeline generalizes to drawings with different
  layouts.
- **Rectangular and diagonal duct detection** — a third detection
  layer anchored on `XX"×YY"` labels.
- **Leader-line following** — handle dimension labels drawn outside
  the duct's bounding region.
- **Full-page interactive UI** — currently the UI uses the cropped
  drawing region. A full-page version would include the title block
  and notes alongside the clickable ducts.

---

## Submission

A short demo video showing the pipeline running and the interactive UI
in use is uploaded separately via the submission portal.