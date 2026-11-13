#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
PYTHON_BIN="$PROJECT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python virtual environment not found: $PYTHON_BIN" >&2
    exit 1
fi

cd "$PROJECT_DIR"
exec "$PYTHON_BIN" "$PROJECT_DIR/GlucosePixelMatrix.py"