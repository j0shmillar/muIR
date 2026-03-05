#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYTEST_PYTHON:-}" ]]; then
  PY="$PYTEST_PYTHON"
elif [[ -x "/opt/anaconda3/envs/py310/bin/python" ]]; then
  PY="/opt/anaconda3/envs/py310/bin/python"
else
  PY="python3"
fi

# Work around a local pytest startup crash in capture initialization.
exec "$PY" -m pytest -p no:capture "$@"
