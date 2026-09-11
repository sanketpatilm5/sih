"""
Write the sample 2D plans and their drawings into `samples/`.

    python scripts/make_samples.py

Each plan is written as JSON (edit this, then upload it) alongside an SVG of
every floor (look at this, to check the shapes are right).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from bhoomi3d.plan2d import write_samples  # noqa: E402

if __name__ == "__main__":
    out = ROOT / "samples"
    written = write_samples(out)
    print(f"Wrote {len(written)} files to {out}")
    for p in written:
        if p.suffix == ".json":
            print(f"  {p.name:<28} <- edit and upload this")
    print("\nUpload via the viewer's '2D -> 3D' tab, or:")
    print("  curl -X POST http://127.0.0.1:8000/api/build-from-plan \\")
    print("       -H 'Content-Type: application/json' \\")
    print("       --data-binary @samples/plan-simple.json")
