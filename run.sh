#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c "import uvicorn, starlette, numpy, scipy, soundfile, multipart" >/dev/null 2>&1; then
  .venv/bin/python -m pip install -r requirements-signaltrace.txt
fi
exec .venv/bin/python -m uvicorn signaltrace_server:app --host 127.0.0.1 --port "${PORT:-8501}"
