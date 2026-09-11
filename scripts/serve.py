"""
Start the Bhoomi3D API and viewer.

    python scripts/serve.py [--port 8000] [--host 127.0.0.1] [--reload]

Then open http://127.0.0.1:8000/ for the viewer, or /docs for the API.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true",
                    help="restart on source changes (development)")
    args = ap.parse_args()

    if not (ROOT / "data" / "derived" / "cadastre.json").exists():
        print("No register found. Build the demonstration dataset first:\n"
              "    python scripts/build_demo.py\n"
              "Starting anyway - the API will report its state at /api/health.\n")

    import uvicorn
    print(f"\n  Viewer   http://{args.host}:{args.port}/")
    print(f"  API docs http://{args.host}:{args.port}/docs\n")
    uvicorn.run("bhoomi3d.api:app", host=args.host, port=args.port,
                reload=args.reload, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
