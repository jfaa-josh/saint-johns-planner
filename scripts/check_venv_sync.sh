#!/usr/bin/env bash
set -euo pipefail
uv sync --check || {
  echo ".venv is out of sync. Run: uv sync"
  exit 1
}
