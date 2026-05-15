# test/inspect_labels.py
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
with open(project_root / "output" / "sample_step2_labels.json") as f:
    labels = json.load(f)

print("All OCR labels with digits, grouped by kind:\n")
for l in sorted(labels, key=lambda x: (x["kind"], x["center_px"][1], x["center_px"][0])):
    if l["source"] != "ocr":
        continue
    if not any(c.isdigit() for c in l["text"]):
        continue
    cx, cy = l["center_px"]
    w, h = l["size"]
    print(f"  {l['kind']:12s} {l['text']!r:20s}  conf={l['conf']:3d}  at ({cx:.0f}, {cy:.0f})  size=({w:.0f}x{h:.0f})")