"""CPU-friendly tiny detector+classifier so the Flask app works before full YOLOv8 training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATASET_DIR, DETECTION_CLASSES, IMAGES_DIR, TINY_WEIGHTS, WEIGHTS_DIR
from src.tiny_model import FractureCNN


class PhantomSet(Dataset):
    def __init__(self, split: str, limit: int | None = None) -> None:
        meta = pd.read_csv(DATASET_DIR / "metadata.csv")
        self.rows = meta[meta["split"] == split].reset_index(drop=True)
        if limit:
            self.rows = self.rows.iloc[:limit]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows.iloc[idx]
        img = cv2.imread(str(IMAGES_DIR / row["file"]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (128, 128)).astype(np.float32) / 255.0
        h0 = w0 = 256
        x1, y1, x2, y2 = [int(v) for v in str(row["bbox_xyxy"]).split(",")]
        cx = ((x1 + x2) / 2) / w0
        cy = ((y1 + y2) / 2) / h0
        bw = (x2 - x1) / w0
        bh = (y2 - y1) / h0
        tensor = torch.from_numpy(img[None, ...])
        label = torch.tensor(int(row["detection_id"]), dtype=torch.long)
        box = torch.tensor([cx, cy, bw, bh], dtype=torch.float32)
        return tensor, label, box


def train(epochs: int, train_limit: int, batch: int, lr: float) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PhantomSet("train", train_limit)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    model = FractureCNN(len(DETECTION_CLASSES)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    l1 = nn.L1Loss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        correct = 0
        seen = 0
        for xb, yb, box in tqdm(loader, desc=f"epoch {epoch+1}/{epochs}"):
            xb, yb, box = xb.to(device), yb.to(device), box.to(device)
            logits, pred_box = model(xb)
            loss = ce(logits, yb) + 2.0 * l1(pred_box, box)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(xb)
            correct += int((logits.argmax(1) == yb).sum().item())
            seen += len(xb)
        print(f"loss={total/seen:.4f} acc={correct/seen:.3f}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": DETECTION_CLASSES}, TINY_WEIGHTS)
    print("Saved", TINY_WEIGHTS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    if not (DATASET_DIR / "metadata.csv").exists():
        raise SystemExit("Run python scripts/build_dataset.py first")
    train(args.epochs, args.limit, args.batch, args.lr)


if __name__ == "__main__":
    main()
