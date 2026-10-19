#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/python -m pip install -r requirements.lock
fi
if [ -z "${AUDIO_LAB_DB:-}" ]; then
  if [ -f data/study.sqlite ]; then
    export AUDIO_LAB_DB=data/study.sqlite
  else
    export AUDIO_LAB_DB=data/demo.sqlite
    if [ ! -f "$AUDIO_LAB_DB" ]; then
      .venv/bin/python -m lab --db "$AUDIO_LAB_DB" demo
    fi
  fi
fi
exec .venv/bin/python -m streamlit run app.py "$@"
