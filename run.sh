#!/bin/sh
set -eu
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python -m sync_demo.app
else
  exec python3 -m sync_demo.app
fi
