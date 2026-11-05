#!/usr/bin/env bash
# TowerMind & AI Gym Tower Defense — Universal Shell Launcher

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# Prefer project virtual environment python if present
if [ -x "$DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$DIR/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

exec "$PYTHON_BIN" "$DIR/run.py" "$@"

