#!/bin/sh
# Start SignalTrace at http://127.0.0.1:8501 (set PORT to change it).
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  PYTHON="${PYTHON:-python3}"
  if ! "$PYTHON" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
    echo "SignalTrace needs Python 3.11 or newer. Rerun with PYTHON=/path/to/python3.13 sh run.sh" >&2
    exit 1
  fi
  "$PYTHON" -m venv .venv
fi
if ! .venv/bin/python -c "import uvicorn, starlette, numpy, scipy, soundfile, multipart" >/dev/null 2>&1; then
  .venv/bin/python -m pip install --disable-pip-version-check -r requirements-signaltrace.txt
fi
exec .venv/bin/python -m uvicorn signaltrace_server:app --host 127.0.0.1 --port "${PORT:-8501}"
