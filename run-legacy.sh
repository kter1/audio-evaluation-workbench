#!/bin/sh
# Retain the original audio annotation and heavy detector interfaces.
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.lock
fi
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 "$@"
