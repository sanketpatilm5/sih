#!/usr/bin/env bash
# One-shot setup and launch for macOS / Linux.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "Creating virtual environment ..."
  python3 -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/python -m pip install -r backend/requirements.txt
fi

if [ ! -f data/derived/cadastre.json ]; then
  echo "Building the demonstration dataset ..."
  ./.venv/bin/python scripts/build_demo.py
fi

echo "Starting Bhoomi3D ..."
exec ./.venv/bin/python scripts/serve.py
