#!/usr/bin/env python3
"""Probe public dataset URLs, count local files, and optionally download via Kaggle.

Usage
-----
  python data/audit_datasets.py                 # local counts + cached remotes
  python data/audit_datasets.py --probe         # HTTP-check every source (no bulk download)
  python data/audit_datasets.py --download bone_break_classification
  python data/audit_datasets.py --download bonefracture_yolo8
  python data/audit_datasets.py --download fracatlas

Kaggle downloads need ~/.kaggle/kaggle.json (Account → Create API Token).
GRAZPEDWRI-DX images are ~15 GB — this script verifies the 20,327-row manifest
by default and only downloads zips if you pass --download grazpedwri_dx.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data"
CATALOG_PATH = DATA / "catalog.json"
REPORT_PATH = DATA / "inventory_report.json"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".dcm"}

CLASS_ALIASES = {
    "avulsion fracture": "Avulsion Fracture",
    "comminuted fracture": "Comminuted Fracture",
    "fracture dislocation": "Fracture-Dislocation",
    "fracture-dislocation": "Fracture-Dislocation",
    "greenstick fracture": "Greenstick Fracture",
    "hairline fracture": "Hairline Fracture",
    "impacted fracture": "Impacted Fracture",
    "longitudinal fracture": "Longitudinal Fracture",
    "oblique fracture": "Oblique Fracture",
    "pathological fracture": "Pathological Fracture",
    "spiral fracture": "Spiral Fracture",
    "elbow positive": "Elbow Positive",
    "fingers positive": "Fingers Positive",
    "forearm fracture": "Forearm Fracture",
    "humerus": "Humerus",
    "humerus fracture": "Humerus Fracture",
    "shoulder fracture": "Shoulder Fracture",
    "wrist positive": "Wrist Positive",
    "pelvic fracture": "Pelvic Fracture",
}


def load_catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text())


def http_get(url: str, timeout: int = 25, limit: int | None = None) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": "FractureNet-DatasetAuditor/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read() if limit is None else resp.read(limit)
        return resp.status, raw.decode("utf-8", errors="replace")


def probe_source(source: dict) -> dict:
    url = source.get("probe") or source.get("url")
    result = {"id": source["id"], "url": source.get("url"), "ok": False, "detail": ""}
    try:
        status, body = http_get(url)
        result["http_status"] = status
        result["ok"] = 200 <= status < 400
        if source["id"].startswith("graz") or source["id"] == "fracatlas":
            meta = json.loads(body) if body.strip().startswith("{") else {}
            # full body may be truncated; refetch if needed
            if "title" not in meta:
                status, raw = http_get(url)
                meta = json.loads(raw) if raw.strip().startswith("{") else {}
            result["title"] = meta.get("title")
            result["n_files"] = len(meta.get("files") or [])
            result["license"] = (meta.get("license") or {}).get("name")
        elif "kaggle.com/api" in url:
            meta = json.loads(body)
            info = meta.get("info") or meta
            result["title"] = info.get("title")
            result["downloads"] = info.get("totalDownloads")
            result["votes"] = info.get("totalVotes")
            result["ok"] = bool(info.get("title"))
        result["detail"] = "reachable"
    except Exception as exc:
        result["detail"] = f"{type(exc).__name__}: {exc}"
        result["ok"] = False
    return result


def count_images(folder: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not folder.exists():
        return counts
    for path in folder.rglob("*"):
        if path.suffix.lower() in IMAGE_EXTS and path.is_file():
            parent = path.parent.name
            key = CLASS_ALIASES.get(parent.lower(), parent)
            counts[key] = counts.get(key, 0) + 1
    return counts


def audit_graz_manifest() -> dict:
    path = DATA / "manifests" / "grazpedwri_dataset.csv"
    if not path.exists():
        return {"present": False, "n_rows": 0}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    fracture = Counter()
    gender = Counter()
    for row in rows:
        fracture[str(row.get("fracture_visible", "")).strip()] += 1
        gender[str(row.get("gender", "")).strip()] += 1
    return {
        "present": True,
        "n_rows": len(rows),
        "expected": 20327,
        "match": len(rows) == 20327,
        "fracture_visible": dict(fracture),
        "gender": dict(gender),
        "path": str(path.relative_to(ROOT)),
    }


def local_inventory() -> dict:
    cls_train = count_images(DATA / "classification" / "train")
    cls_val = count_images(DATA / "classification" / "val")
    yolo_img = 0
    yolo_lbl = 0
    yolo_root = DATA / "yolo"
    if yolo_root.exists():
        yolo_img = sum(1 for p in (yolo_root / "images").rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        yolo_lbl = sum(1 for p in (yolo_root / "labels").rglob("*.txt") if p.is_file())
    return {
        "classification_train": cls_train,
        "classification_val": cls_val,
        "classification_train_total": sum(cls_train.values()),
        "classification_val_total": sum(cls_val.values()),
        "yolo_images": yolo_img,
        "yolo_labels": yolo_lbl,
        "graz": audit_graz_manifest(),
        "fracatlas_images": sum(count_images(DATA / "fracatlas").values()),
    }


def class_checklist(catalog: dict, local: dict) -> list[dict]:
    paper = next(s for s in catalog["sources"] if s["id"] == "bone_break_classification")
    rows = []
    train = local["classification_train"]
    val = local["classification_val"]
    for name, expected in paper["class_counts_paper"].items():
        have = train.get(name, 0) + val.get(name, 0)
        rows.append(
            {
                "class": name,
                "expected": expected,
                "local": have,
                "status": "ok" if have >= expected else ("partial" if have else "missing"),
            }
        )
    return rows


def kaggle_download(slug: str, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    token = Path.home() / ".kaggle" / "kaggle.json"
    if not token.exists() and not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        raise SystemExit(
            "Kaggle credentials missing. Create an API token at kaggle.com/settings "
            "and save it as ~/.kaggle/kaggle.json (chmod 600)."
        )
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    print("Running", " ".join(cmd))
    subprocess.check_call(cmd)


def download_fracatlas(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "FracAtlas.zip"
    url = "https://ndownloader.figshare.com/files/40344400"
    # Resolve actual file URL from API
    status, body = http_get("https://api.figshare.com/v2/articles/22363012")
    meta = json.loads(body)
    files = meta.get("files") or []
    if files:
        url = files[0].get("download_url") or url
    print(f"Downloading FracAtlas (~323 MB) from {url}")
    urllib.request.urlretrieve(url, zip_path)
    subprocess.check_call(["unzip", "-o", str(zip_path), "-d", str(dest)])


def build_report(probe: bool) -> dict:
    catalog = load_catalog()
    local = local_inventory()
    remotes = []
    if probe:
        for source in catalog["sources"]:
            print(f"Probing {source['id']} …")
            remotes.append(probe_source(source))
    elif REPORT_PATH.exists():
        prev = json.loads(REPORT_PATH.read_text())
        remotes = prev.get("remotes") or []
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "probed": probe,
        "scale_up_math": catalog["scale_up_math"],
        "remotes": remotes,
        "local": local,
        "morphology_checklist": class_checklist(catalog, local),
        "how_to_download": {
            "kaggle_token": "https://www.kaggle.com/settings → API → Create New Token → ~/.kaggle/kaggle.json",
            "paper_10_class": "kaggle datasets download -d pkdarabi/bone-break-classification-image-dataset -p data/raw --unzip",
            "abstract_yolo": "kaggle datasets download -d pkdarabi/bone-fracture-detection-computer-vision-project -p data/raw --unzip",
            "graz_20k_images": "kaggle datasets download -d jasonroggy/grazpedwri-dx -p data/raw  # ~16 GB",
            "fracatlas": "python data/audit_datasets.py --download fracatlas",
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report


def print_summary(report: dict) -> None:
    print("\n=== Remote availability ===")
    if not report.get("remotes"):
        print("Run with --probe to HTTP-check Kaggle / Figshare (no image download).")
    for remote in report.get("remotes") or []:
        flag = "LIVE" if remote.get("ok") else "DOWN"
        print(f"  [{flag}] {remote['id']}: {remote.get('title') or remote.get('detail')}")
    print("\n=== Local images ===")
    local = report["local"]
    print(f"  classification train: {local['classification_train_total']}")
    print(f"  classification val:   {local['classification_val_total']}")
    print(f"  YOLO images/labels:   {local['yolo_images']} / {local['yolo_labels']}")
    graz = local["graz"]
    print(f"  GRAZPEDWRI-DX manifest rows: {graz.get('n_rows')} (expected 20327, match={graz.get('match')})")
    print("\n=== 10 morphology classes vs paper Table 1 ===")
    for row in report["morphology_checklist"]:
        print(f"  {row['status']:8} {row['class']:24} local={row['local']:4} expected={row['expected']}")
    math = report["scale_up_math"]
    print(
        f"\n20k scale: GRAZ {math['grazpedwri_dx_images']} + YOLO {math['abstract_yolo_images']} "
        f"+ FracAtlas {math['fracatlas_images']} + paper {math['paper_morphology_images']} "
        f"≈ {math['combined_unique_estimate']}+ radiographs (not 20k of every morphology class)."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit / download fracture datasets")
    parser.add_argument("--probe", action="store_true", help="HTTP-check every catalog URL")
    parser.add_argument(
        "--download",
        choices=["bone_break_classification", "bonefracture_yolo8", "fracatlas", "grazpedwri_dx"],
        help="Download one source (Kaggle token required except fracatlas)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.download == "bone_break_classification":
        kaggle_download("pkdarabi/bone-break-classification-image-dataset", DATA / "raw" / "bone_break")
    elif args.download == "bonefracture_yolo8":
        kaggle_download("pkdarabi/bone-fracture-detection-computer-vision-project", DATA / "raw" / "yolo")
    elif args.download == "grazpedwri_dx":
        kaggle_download("jasonroggy/grazpedwri-dx", DATA / "raw" / "grazpedwri")
    elif args.download == "fracatlas":
        download_fracatlas(DATA / "raw" / "fracatlas")
    report = build_report(probe=args.probe or not REPORT_PATH.exists())
    print_summary(report)
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
