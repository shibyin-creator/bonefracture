"""Orthopedic post-treatment simulation: alignment vectors + implant overlay.

Accepts a radiograph and optional YOLO/Grad-CAM box. Computes a fracture-gap
realignment vector, warps proximal/distal fragments toward reduction, then
blends metallic plates, cortical screws, or an IM rod with OpenCV.

Returns a side-by-side panel:
Pre-Operative Dislocated Fracture | Post-Operative Simulated Surgical Re-fixation

Research & diagnostic support only — not an authorized medical device.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from refix_simulation.simulator import (
    HardwarePlan,
    RefixResult,
    save_refixation,
    simulate_refixation,
)


def alignment_vector(box: tuple[int, int, int, int], magnitude: float = 12.0) -> tuple[float, float]:
    """Vector along the box long axis used to close a simulated fracture gap."""
    x1, y1, x2, y2 = box
    dx, dy = float(x2 - x1), float(y2 - y1)
    n = max((dx * dx + dy * dy) ** 0.5, 1.0)
    return (magnitude * dx / n, magnitude * dy / n)


def render_refixation(
    image_bgr: np.ndarray,
    predicted_class: str,
    box: tuple[int, int, int, int] | None = None,
    output_dir: Path | None = None,
    stem: str = "refix",
) -> dict:
    """Run realignment + hardware overlay and optionally write PNG files."""
    result: RefixResult = simulate_refixation(image_bgr, predicted_class, box=box)
    payload = {
        "hardware": result.plan.hardware,
        "note": result.plan.note,
        "gap_vector": result.gap_vector,
        "disclaimer": "Research & Diagnostic Support Tool Only — Not an Authorized Medical Device",
    }
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pre = output_dir / f"{stem}_pre_op.png"
        post = output_dir / f"{stem}_post_op.png"
        side = output_dir / f"{stem}_side_by_side.png"
        cv2.imwrite(str(pre), result.pre_op)
        cv2.imwrite(str(post), result.post_op)
        cv2.imwrite(str(side), result.side_by_side)
        payload["pre_op"] = str(pre)
        payload["post_op"] = str(post)
        payload["side_by_side"] = str(side)
    return payload


__all__ = [
    "HardwarePlan",
    "RefixResult",
    "alignment_vector",
    "render_refixation",
    "save_refixation",
    "simulate_refixation",
]
