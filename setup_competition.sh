#!/usr/bin/env bash
set -euo pipefail

KEY_PATH="${1:-competition_secrets/gee-service-account-key.json}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
website_dir="$repo_root/website"
backend_dir="$website_dir/backend"
source_key="$repo_root/$KEY_PATH"
target_key="$backend_dir/gee-service-account-key.json"

if [[ ! -f "$source_key" ]]; then
  echo
  echo "Missing Earth Engine key file."
  echo "Expected at: $source_key"
  echo "Place the competition key there, then run this script again."
  exit 1
fi

echo "Copying Earth Engine key to backend runtime path..."
cp "$source_key" "$target_key"

cd "$website_dir"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "Creating virtual environment..."
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv .venv
  elif command -v python >/dev/null 2>&1; then
    python -m venv .venv
  else
    echo "Python launcher not found. Install Python 3.10+."
    exit 1
  fi
fi

venv_python=".venv/bin/python"
if [[ ! -x "$venv_python" ]]; then
  echo "Virtual environment python not found at $venv_python"
  exit 1
fi

if [[ "$SKIP_INSTALL" != "1" ]]; then
  echo "Installing backend dependencies..."
  "$venv_python" -m pip install -r requirements.txt
fi

echo
echo "Starting website backend, and will be shown once loaded in at http://127.0.0.1:8000. NOT READY YET"
exec "$venv_python" -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
