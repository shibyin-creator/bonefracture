#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/OsteoScan_BoneFractureCAD_20k.zip}"
cd "$ROOT"
if [[ ! -d dataset/images/train ]]; then
  python3 scripts/build_dataset.py --images 20000
fi
zip -r -q "$OUT" . \
  -x '.git/*' -x '.git/**' \
  -x '*/__pycache__/*' -x '*.pyc' \
  -x '.venv/*' -x 'uploads/*' \
  -x '*.zip'
echo "Wrote $OUT"
ls -lh "$OUT"
EOF