# HVAC Duct Detection and Annotation

A pipeline that ingests an HVAC mechanical drawing (PDF), detects round
horizontal ducts, traces U-shaped connections between them, and produces
an annotated image plus structured per-duct metadata (dimension, pressure
class, area, position, length) for an interactive UI.

> **Status:** Steps 1–4 complete. The interactive UI (Step 5), final
> packaging, and demo recording are in progress.

---

## Pipeline overview

```
input/sample.pdf
    │
    ▼
Step 1 — load & crop      (steps/load.py)
    │   Rasterizes the PDF at 300 DPI and crops to the drawing region
    │
    ▼
Step 2 — labels            (steps/labels.py)
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
    │   For each detected duct, resolves:
    │     • Dimension     (e.g. 14"ɸ) via directed OCR on a small crop just
    │                     left of the ɸ anchor, with Step 2 labels as a
    │                     fallback.
    │     • Pressure class (Low / Medium / High) via a diameter-based
    │                     heuristic, documented under "Assumptions" below.
    │     • Area          (Kitchen / Dining / Scullery / …) by Euclidean
    │                     distance from the duct centroid to the bounding
    │                     box of the nearest visible room-name text.
    │     • Position      Two fields: position_x (left/center/right) and
    │                     position_y (top/middle/bottom), bucketed against
    │                     adaptive bounds derived from all duct centroids.
    │     • Length        Centerline length in feet, using the drawing
    │                     scale (1/4" = 1'-0" at 300 DPI → 75 px/ft).
    │     • Orientation   "horizontal" or "vertical".
    │     • Click bboxes  Hit areas around each centerline so the UI can
    │                     map a click to a duct.
    │
    │   U-pair handling: each U-shape from Layer 2 becomes THREE separate
    │   duct entries (top horizontal, vertical connector, bottom horizontal).
    │   Verticals have no labeled dimension in plan view, but they inherit
    │   the pressure class of the horizontals they connect, since the air
    │   in all three segments belongs to the same physical run.
    │
    ▼
output/sample_step3_layer2.png   (annotated drawing)
output/sample_step4_ducts.json   (per-duct metadata for the UI)
output/sample_step4_ducts.csv    (flat table view)
```

---

## Assumptions

A few deliberate simplifications and design choices were made. All are
visible in `steps/extract_metadata.py` and tunable from the KNOBS block at
the top of that file.

- **Pressure class** is derived from a size-based heuristic, not from the
  drawing's seal class or equipment schedule (those would normally come
  from a different sheet or a SMACNA notation that isn't present here):
  - diameter ≤ 8" → **Low**
  - diameter 9–14" → **Medium**
  - diameter ≥ 15" → **High**

  Real-world classification depends on static pressure (in. w.g.) and
  velocity, which the drawing doesn't state directly. The thresholds
  above are a defensible proxy.

- **Round-duct dimensions** are resolved by directed OCR on a small crop
  immediately left of each ɸ symbol detected in Layer 1, with Step 2's
  general-page OCR results as a fallback. Confidence is ~90% on this
  sample; misses are recorded as `null`.

- **Vertical-leg dimension and pressure.** Vertical connectors in U-shape
  routings carry no labeled ɸ dimension in plan view (the symbol is drawn
  on the horizontal portions). Vertical entries therefore have
  `dimension: null`. Pressure class is inherited from the horizontals of
  the same U-pair, since the air in all three segments belongs to the
  same physical run.

- **Area assignment** uses Euclidean distance from the duct centroid to
  the bounding box of the nearest *visible* room-name text on the drawing
  (e.g. `KITCHEN`, `DINING`, `SCULLERY`). Distance to a bbox is zero if
  the centroid is inside the bbox, otherwise the minimum straight-line
  distance to the nearest edge. Only OCR-source room names are used —
  PDF-source room names can come from rotated text in schedules or
  legends at coordinates that don't match where the rooms actually are
  visually. Two room labels (`WALK-IN COOLER`, `WALK-IN FREEZER`) appear
  only in PDF source and are therefore not used; no duct in the sample
  drawing is closest to either of those rooms anyway.

  Because the room-name text is a single anchor per room, a long duct
  that crosses a room boundary may be assigned to the smaller adjacent
  room if that room's name text happens to sit closer to the duct's
  centroid. This is documented as a known characteristic rather than a
  bug.

- **Position** is reported as two independent buckets: `position_x` in
  {left, center, right} and `position_y` in {top, middle, bottom}. The
  bands are computed *adaptively* — the min/max x,y of all duct centroids
  define the active region, and that region is split into thirds. This
  avoids a misleading "middle" classification just because the rendered
  PDF has a lot of empty margin around the drawing.

- **Centerline length** is converted to feet using the drawing scale:
  300 DPI raster, drawing scale `1/4" = 1'-0"`, giving 75 px/ft. Values
  are rounded to whole feet for display.

- **Click hit-areas** are axis-aligned bounding boxes around each duct
  centerline (±35 px). Generous enough that clicks near a duct still
  register; hit-areas for nearby parallel ducts may overlap, in which
  case the first matching duct in the data is returned.

---

## Known limitations

The current detection is tuned to the provided sample drawing. Two
limitations to be aware of:

### 1. Hard-coded anchor points and selection criteria

Several parameters in the detection pipeline are hard-coded to coordinates
from the sample drawing:

- **Layer 1** uses a fixed template location `(2942, 1204)` to extract
  the ɸ-symbol reference image. Cropping the drawing region in Step 1
  also uses hard-coded pixel bounds.
- **Layer 2** selects which duct pairs to trace by absolute pixel ranges
  (e.g. *"dining pair = ducts with `left_x` between 4800 and 5100"*).
  This works for the sample but will not generalize to a new drawing
  without re-tuning the ranges.

A more general approach would derive these dynamically — for example,
detect the title block to locate the drawing region, and cluster Layer 1
detections by their endpoint coordinates to discover U-pair candidates
without absolute filters. Out of scope for this submission.

### 2. Angular and rectangular ducts are not detected

The pipeline targets **round horizontal ducts** only (those labeled with
the ɸ diameter symbol). Two duct types are not detected:

- **Rectangular ducts** labeled as `XX"×YY"` (e.g. the `22"×14"` supply
  between the kitchen and the dining/service area). These use a different
  labeling convention and would need a separate detection layer.
- **Angled / diagonal ducts** (e.g. the same `22"×14"` segment, which
  runs at roughly 30–45° rather than horizontally). The horizontal-scan
  logic in Layer 1 and the orthogonal wall-tracing in Layer 2 both assume
  axis-aligned ducts.

These would be a logical Layer 4 — anchor on the `XX"×YY"` label via OCR,
then search outward for two parallel dark lines at arbitrary angle.

### 3. Very short stubs may have unknown dimension

One duct in the sample (a 152-px connector segment near the bottom of the
kitchen area) has no recoverable dimension because the ɸ label is not
present in the directed-OCR crop region and no nearby fallback label
matched. This duct appears in the output with `dimension: null` and is
shown as "Unknown" in the UI. It remains clickable and reports its
geometry (orientation, area, position, length) like any other duct.

---

## Repository layout

```
duct_tool/
├── input/          ← place sample.pdf here
├── output/         ← all generated files
├── steps/
│   ├── load.py
│   ├── labels.py
│   ├── detect_layer_1.py
│   ├── detect_layer_2.py
│   └── extract_metadata.py
├── requirements.txt
└── README.md
```

---

## How to run

> _Detailed setup instructions will be added once the full pipeline
> (Step 4 + UI) is in place._