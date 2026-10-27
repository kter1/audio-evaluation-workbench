#!/bin/sh
# Start the blind annotation workspace (Streamlit) at http://127.0.0.1:8501.
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  PYTHON="${PYTHON:-python3}"
  if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    echo "The annotation workspace needs Python 3.11 or newer. Rerun with PYTHON=/path/to/python3.13 sh run-legacy.sh" >&2
    exit 1
  fi
  "$PYTHON" -m venv .venv
fi
if ! .venv/bin/python -c "import streamlit, pandas, plotly" >/dev/null 2>&1; then
  .venv/bin/python -m pip install --disable-pip-version-check -r requirements.lock
fi
if [ -z "${AUDIO_LAB_DB:-}" ] && [ ! -f data/challenge-v4.sqlite ] && [ ! -f data/study.sqlite ] && [ ! -f data/demo.sqlite ]; then
  .venv/bin/python -m lab --db data/demo.sqlite demo
fi
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 "$@"
