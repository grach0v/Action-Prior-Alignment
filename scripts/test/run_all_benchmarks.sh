#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-"$ROOT/.venv/bin/python"}
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN=python3
fi

exec "$PYTHON_BIN" "$ROOT/scripts/test/run_all_benchmarks.py" "$@"
