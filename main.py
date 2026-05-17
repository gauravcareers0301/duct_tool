# main.py
"""
Pipeline runner: executes all six pipeline steps in order.

Usage (from project root):
    python main.py

After this finishes successfully, launch the interactive UI with:
    streamlit run App.py
"""
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
STEPS_DIR    = PROJECT_ROOT / "steps"

# Order matters: each step consumes outputs from previous steps.
STEPS = [
    ("Step 1 -- Load & crop PDF",            "load.py"),
    ("Step 2 -- Extract labels (OCR + PDF)", "labels.py"),
    ("Step 3 -- Detect ducts (Layer 1)",     "detect_layer_1.py"),
    ("Step 3 -- Trace U-shapes (Layer 2)",   "detect_layer_2.py"),
    ("Step 4 -- Extract per-duct metadata",  "extract_metadata.py"),
    ("Step 6 -- Compose final deliverable",  "compose_final.py"),
]


def run_step(label, script_name):
    script_path = STEPS_DIR / script_name
    if not script_path.exists():
        print(f"\n  MISSING: {script_path}")
        return False

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  > python {script_path.relative_to(PROJECT_ROOT)}")
    print(f"{'=' * 70}")

    t0 = time.time()
    # Use the same Python interpreter (so the venv is honored).
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=PROJECT_ROOT,
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  FAILED after {elapsed:.1f}s (exit code {result.returncode})")
        return False
    print(f"\n  OK ({elapsed:.1f}s)")
    return True


def main():
    # Sanity check: is there a PDF to process?
    input_dir = PROJECT_ROOT / "input"
    if not input_dir.exists() or not list(input_dir.glob("*.pdf")):
        print(f"ERROR: no PDF found in {input_dir.relative_to(PROJECT_ROOT)}/")
        print("Place an HVAC drawing PDF in the input/ folder and try again.")
        sys.exit(1)

    t_start = time.time()
    print(f"\nRunning HVAC duct detection pipeline from {PROJECT_ROOT}")

    for label, script in STEPS:
        if not run_step(label, script):
            print(f"\nPipeline halted at: {label}")
            sys.exit(1)

    total = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"  Pipeline complete in {total:.1f}s")
    print(f"{'=' * 70}")
    print("\nOutputs are in output/. Next step:")
    print("    streamlit run App.py")


if __name__ == "__main__":
    main()