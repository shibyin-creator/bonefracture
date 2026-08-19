"""Orthopedic hardware overlay for post-surgical refixation demonstration.

Severe / displaced detections (comminuted, dislocation, spiral, etc.) trigger
OpenCV blending of plates, screws, and intramedullary rods onto the X-ray.
This is an educational simulation, not a surgical planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class HardwarePlan:
    hardware: str
    note: str
    color: tuple[int, int, int]


PLANS: dict[str, HardwarePlan] = {
    "Comminuted Fracture": HardwarePlan(
        "locking compression plate + cortical screws",
        "Multi-fragment injury: bridge plating restores length, alignment, and rotation.",
        (40, 210, 230),
    ),
    "Fracture-Dislocation": HardwarePlan(
        "anatomical plate + reduction screws",
        "Joint reduction followed by rigid plate fixation across the articular surface.",
        (80, 180, 255),
    ),
    "Spiral Fracture": HardwarePlan(
        "intramedullary interlocking nail",
        "Long spiral pattern: IM rod with proximal and distal interlocking screws.",
        (70, 220, 160),
    ),
    "Impacted Fracture": HardwarePlan(
        "buttress plate + lag screws",
        "Impacted metaphysis: buttress plate prevents collapse after elevation.",
        (90, 200, 255),
    ),
    "Pathological Fracture": HardwarePlan(
        "prophylactic IM nail + cement augmentation",
        "Weakened bone: load-sharing nail with optional cement around the lesion.",
        (160, 140, 255),
    ),
    "Humerus Fracture": HardwarePlan(
        "anterolateral plate or IM humeral nail",
        "Humeral shaft: plate osteosynthesis or locked IM nailing.",
        (50, 200, 220),
    ),
    "Forearm Fracture": HardwarePlan(
        "dual 3.5 mm compression plates",
        "Radius and ulna: independent compression plates to restore radial bow.",
        (40, 190, 240),
    ),
    "Shoulder Fracture": HardwarePlan(
        "proximal humerus locking plate",
        "Proximal humerus: PHILOS-style locking plate with calcar screws.",
        (70, 170, 255),
    ),
    "Pelvic Fracture": HardwarePlan(
        "sacroiliac screws + reconstruction plate",
        "Pelvic ring: percutaneous SI screws and anterior reconstruction plate.",
        (90, 160, 230),
    ),
}

DEFAULT_PLAN = HardwarePlan(
    "neutralization plate + lag screws",
    "Standard AO fixation: lag screw compression neutralized by a protection plate.",
    (50, 200, 220),
)


def _bone_axis(gray: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """Estimate a long-bone axis from the brightest cortical structures."""
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = gray.shape
        return (w // 2, int(h * 0.18)), (w // 2, int(h * 0.82)), (w // 2, h // 2)
    cnt = max(contours, key=cv2.contourArea)
    line = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    vx, vy, x, y = (float(v) for v in line[:4])
    h, w = gray.shape
    length = 0.38 * max(h, w)
    p1 = (int(x - vx * length), int(y - vy * length))
    p2 = (int(x + vx * length), int(y + vy * length))
    center = (int(x), int(y))
    return p1, p2, center


def _clip(pt: tuple[int, int], shape: tuple[int, ...]) -> tuple[int, int]:
    h, w = shape[:2]
    return max(0, min(w - 1, pt[0])), max(0, min(h - 1, pt[1]))


def _draw_plate(canvas: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], color: tuple[int, int, int]) -> None:
    overlay = canvas.copy()
    cv2.line(overlay, p1, p2, color, thickness=18, lineType=cv2.LINE_AA)
    cv2.line(overlay, p1, p2, (230, 230, 230), thickness=8, lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, dst=canvas)


def _draw_screws(canvas: np.ndarray, p1: tuple[int, int], p2: tuple[int, int], n: int = 6) -> None:
    for i in range(n):
        t = 0.08 + 0.84 * i / max(n - 1, 1)
        x = int(p1[0] + t * (p2[0] - p1[0]))
        y = int(p1[1] + t * (p2[1] - p1[1]))
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1.0)
        nx, ny = -dy / length, dx / length
        a = (int(x + nx * 22), int(y + ny * 22))
        b = (int(x - nx * 22), int(y - ny * 22))
        cv2.line(canvas, a, b, (210, 215, 220), 3, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 4, (40, 40, 40), -1, cv2.LINE_AA)


def _draw_rod(canvas: np.ndarray, p1: tuple[int, int], p2: tuple[int, int]) -> None:
    overlay = canvas.copy()
    cv2.line(overlay, p1, p2, (190, 195, 200), 10, cv2.LINE_AA)
    cv2.circle(overlay, p1, 7, (160, 170, 180), -1, cv2.LINE_AA)
    cv2.circle(overlay, p2, 7, (160, 170, 180), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, dst=canvas)


def simulate_refixation(
    image_bgr: np.ndarray,
    predicted_class: str,
    box: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, HardwarePlan]:
    """Blend educational fixation hardware onto a copy of the radiograph."""
    plan = PLANS.get(predicted_class, DEFAULT_PLAN)
    canvas = image_bgr.copy()
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    if box is not None:
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        p1 = _clip((cx, y1), canvas.shape)
        p2 = _clip((cx, y2), canvas.shape)
        center = (cx, cy)
    else:
        p1, p2, center = _bone_axis(gray)
        p1, p2, center = _clip(p1, canvas.shape), _clip(p2, canvas.shape), _clip(center, canvas.shape)

    if "nail" in plan.hardware or "IM" in plan.hardware or "rod" in plan.hardware:
        _draw_rod(canvas, p1, p2)
        _draw_screws(canvas, p1, p2, n=4)
    else:
        _draw_plate(canvas, p1, p2, plan.color)
        _draw_screws(canvas, p1, p2, n=6)

    cv2.circle(canvas, center, 10, (0, 255, 180), 2, cv2.LINE_AA)
    banner = f"REFIXATION DEMO  |  {plan.hardware}"
    cv2.rectangle(canvas, (8, 8), (min(canvas.shape[1] - 8, 8 + 12 * len(banner)), 42), (10, 28, 36), -1)
    cv2.putText(canvas, banner, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 240), 1, cv2.LINE_AA)
    return canvas, plan


def save_refixation(image_bgr: np.ndarray, predicted_class: str, destination: Path, box=None) -> tuple[Path, HardwarePlan]:
    overlay, plan = simulate_refixation(image_bgr, predicted_class, box=box)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), overlay)
    return destination, plan
