"""Public dataset catalog — delegates to data/audit_datasets.py."""

from __future__ import annotations

import json

from config import Config


def probe_and_report(*, probe: bool = True) -> dict:
    from data.audit_datasets import build_report, print_summary

    report = build_report(probe=probe)
    print_summary(report)
    return report


def load_catalog() -> dict:
    path = Config.DATASET_DIR / "catalog.json"
    return json.loads(path.read_text())
