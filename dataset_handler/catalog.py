"""Public dataset catalog, live URL probe, and GRAZPEDWRI-DX 20,327-row audit."""

from __future__ import annotations

import json
import sys

from config import Config, ROOT


def probe_and_report(*, probe: bool = True) -> dict:
    legacy = ROOT / "bone_fracture_system"
    if str(legacy) not in sys.path:
        sys.path.insert(0, str(legacy))
    from data.audit_datasets import build_report, print_summary

    report = build_report(probe=probe)
    print_summary(report)
    return report


def load_catalog() -> dict:
    path = Config.DATASET_DIR / "catalog.json"
    return json.loads(path.read_text())
