#!/usr/bin/env bash
# Paralic launcher for macOS / Linux: creates a virtual environment on first
# run, installs the dependencies, then starts the server and opens the browser.
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  for candidate in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then PYTHON="$candidate"; break; fi
  done
fi
if [ -z "$PYTHON" ]; then
  echo "Python 3.9 or newer is required: https://www.python.org/downloads/" >&2
  exit 1
fi
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' || {
  echo "Python 3.9 or newer is required (found $("$PYTHON" --version 2>&1))." >&2
  exit 1
}

if [ ! -x .venv/bin/python ]; then
  echo "Creating virtual environment (.venv)…"
  "$PYTHON" -m venv .venv
fi
if [ ! -f .venv/.installed ] || [ requirements.txt -nt .venv/.installed ]; then
  echo "Installing dependencies (first run only, this can take a minute)…"
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.txt
  touch .venv/.installed
fi

exec .venv/bin/python -m paralic "$@"
