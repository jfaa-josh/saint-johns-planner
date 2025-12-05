#!/usr/bin/env bash
set -euo pipefail

PY_VER_FILE=$(cut -d. -f1,2 .python-version)

# Get the first requires-python line, then extract the FIRST X.Y on that line
REQ_LINE=$(grep -E '^\s*requires-python\s*=' pyproject.toml | head -n1)
PY_VER_TOML=$(printf '%s\n' "$REQ_LINE" | grep -oE '[0-9]+\.[0-9]+' | head -n1)

if [[ -z "${PY_VER_TOML:-}" ]]; then
  echo "Could not parse requires-python from pyproject.toml"
  exit 1
fi

if [[ "$PY_VER_FILE" != "$PY_VER_TOML" ]]; then
  echo "Version mismatch: .python-version = $PY_VER_FILE but pyproject.toml = $PY_VER_TOML"
  exit 1
fi

echo "Python version match confirmed: $PY_VER_FILE"
