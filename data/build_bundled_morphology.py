"""Write a committed 10-class morphology demo set so training runs without Kaggle."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from dataset_handler.preprocess import make_sample_structure  # noqa: E402


def main() -> None:
    dest = ROOT / "data" / "bundled_morphology"
    make_sample_structure(dest, per_class=8)
    print("Wrote", dest)


if __name__ == "__main__":
    main()
