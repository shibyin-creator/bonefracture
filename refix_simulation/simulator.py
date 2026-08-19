"""Orthopedic re-fixation simulation: gap realignment + hardware overlay.

Educational visualization only — not a surgical planner or implant sizer.

Gap vector: if a YOLO box is present, the fracture gap is modelled as the
short axis of the box. Fragments are translated by ±0.5·gap along the
bone long axis to illustrate reduction, then plates/screws/IM rods are
blended with OpenCV.
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


@dataclass
class RefixResult:
    pre_op: np.ndarray
    reduced: np.ndarray
    post_op: np.ndarray
    side_by_side: np.ndarray
    plan: HardwarePlan
    gap_vector: tuple[float, float]


PLANS: dict[str, HardwarePlan] = {
    "Comminuted Fracture": HardwarePlan("locking compression plate + cortical screws", "Bridge plating restores length, alignment, and rotation.", (40, 210, 230)),
    "Fracture-Dislocation": HardwarePlan("anatomical plate + reduction screws", "Joint reduction then rigid plate fixation.", (80, 180, 255)),
    "Spiral Fracture": HardwarePlan("intramedullary interlocking nail", "IM rod with proximal and distal interlocking screws.", (70, 220, 160)),
    "Impacted Fracture": HardwarePlan("buttress plate + lag screws", "Buttress plate prevents metaphyseal collapse.", (90, 200, 255)),
    "Pathological Fracture": HardwarePlan("prophylactic IM nail + cement", "Load-sharing nail around a weak lesion.", (160, 140, 255)),
    "Humerus Fracture": HardwarePlan("anterolateral plate or IM humeral nail", "Plate osteosynthesis or locked nailing.", (50, 200, 220)),
    "Forearm Fracture": HardwarePlan("dual 3.5 mm compression plates", "Independent plates restore radial bow.", (40, 190, 240)),
    "Shoulder Fracture": HardwarePlan("proximal humerus locking plate", "PHILOS-style plate with calcar screws.", (70, 170, 255)),
    "Pelvic Fracture": HardwarePlan("sacroiliac screws + reconstruction plate", "Percutaneous SI screws and anterior plate.", (90, 160, 230)),
    "Femur": HardwarePlan("cephalomedullary nail", "Femoral reconstruction with interlocking nail.", (60, 190, 210)),
    "Spine": HardwarePlan("pedicle-screw rod construct", "Posterior instrumentation spanning the injured levels.", (90, 160, 220)),
}

DEFAULT_PLAN = HardwarePlan("neutralization plate + lag screws", "AO lag-screw compression neutralized by a protection plate.", (50, 200, 220))


def _clip(pt: tuple[int, int], shape: tuple[int, ...]) -> tuple[int, int]:
    h, w = shape[:2]
    return max(0, min(w - 1, pt[0])), max(0, min(h - 1, pt[1]))


def bone_axis(gray: np.ndarray, box: tuple[int, int, int, int] | None = None):
    if box is not None:
        x1, y1, x2, y2 = box
        p1 = (int((x1 + x2) / 2), y1)
        p2 = (int((x1 + x2) / 2), y2)
        return p1, p2, (int((x1 + x2) / 2), int((y1 + y2) / 2))
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    if not contours:
        return (w // 2, int(h * 0.18)), (w // 2, int(h * 0.82)), (w // 2, h // 2)
    cnt = max(contours, key=cv2.contourArea)
    line = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    vx, vy, x, y = (float(v) for v in line[:4])
    length = 0.38 * max(h, w)
    p1 = (int(x - vx * length), int(y - vy * length))
    p2 = (int(x + vx * length), int(y + vy * length))
    return p1, p2, (int(x), int(y))


def gap_vector(p1: tuple[int, int], p2: tuple[int, int], magnitude: float = 12.0) -> tuple[float, float]:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    n = max((dx * dx + dy * dy) ** 0.5, 1.0)
    return (magnitude * dx / n, magnitude * dy / n)


def realign_fragments(image_bgr: np.ndarray, box: tuple[int, int, int, int] | None, vector: tuple[float, float]) -> np.ndarray:
    """Translate distal and proximal halves toward each other (simulated reduction)."""
    h, w = image_bgr.shape[:2]
    if box is None:
        cy = h // 2
        x1, y1, x2, y2 = 0, 0, w, h
    else:
        x1, y1, x2, y2 = box
        cy = (y1 + y2) // 2
    vx, vy = vector
    reduced = image_bgr.copy()
    # proximal (top) shift + distal (bottom) opposite shift
    m_top = np.float32([[1, 0, vx / 2], [0, 1, vy / 2]])
    m_bot = np.float32([[1, 0, -vx / 2], [0, 1, -vy / 2]])
    top = cv2.warpAffine(image_bgr, m_top, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    bot = cv2.warpAffine(image_bgr, m_bot, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    reduced[:cy] = top[:cy]
    reduced[cy:] = bot[cy:]
    return reduced


def _draw_plate(canvas, p1, p2, color) -> None:
    overlay = canvas.copy()
    cv2.line(overlay, p1, p2, color, 18, cv2.LINE_AA)
    cv2.line(overlay, p1, p2, (230, 230, 230), 8, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, dst=canvas)


def _draw_screws(canvas, p1, p2, n: int = 6) -> None:
    for i in range(n):
        t = 0.08 + 0.84 * i / max(n - 1, 1)
        x = int(p1[0] + t * (p2[0] - p1[0]))
        y = int(p1[1] + t * (p2[1] - p1[1]))
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1.0)
        nx, ny = -dy / length, dx / length
        cv2.line(canvas, (int(x + nx * 22), int(y + ny * 22)), (int(x - nx * 22), int(y - ny * 22)), (210, 215, 220), 3, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 4, (40, 40, 40), -1, cv2.LINE_AA)


def _draw_rod(canvas, p1, p2) -> None:
    overlay = canvas.copy()
    cv2.line(overlay, p1, p2, (190, 195, 200), 10, cv2.LINE_AA)
    cv2.circle(overlay, p1, 7, (160, 170, 180), -1, cv2.LINE_AA)
    cv2.circle(overlay, p2, 7, (160, 170, 180), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, dst=canvas)


def simulate_refixation(
    image_bgr: np.ndarray,
    predicted_class: str,
    box: tuple[int, int, int, int] | None = None,
) -> RefixResult:
    plan = PLANS.get(predicted_class, DEFAULT_PLAN)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    p1, p2, center = bone_axis(gray, box)
    p1, p2, center = _clip(p1, image_bgr.shape), _clip(p2, image_bgr.shape), _clip(center, image_bgr.shape)
    gv = gap_vector(p1, p2)
    reduced = realign_fragments(image_bgr, box, gv)
    post = reduced.copy()
    if "nail" in plan.hardware or "IM" in plan.hardware or "rod" in plan.hardware or "pedicle" in plan.hardware:
        _draw_rod(post, p1, p2)
        _draw_screws(post, p1, p2, n=4)
    else:
        _draw_plate(post, p1, p2, plan.color)
        _draw_screws(post, p1, p2, n=6)
    cv2.putText(post, f"POST-OP  {plan.hardware}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 240), 1, cv2.LINE_AA)
    pre = image_bgr.copy()
    cv2.putText(pre, "PRE-OP FRACTURED X-RAY", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 200, 210), 1, cv2.LINE_AA)
    gap = 12
    h = max(pre.shape[0], post.shape[0])
    canvas = np.zeros((h, pre.shape[1] + post.shape[1] + gap, 3), dtype=np.uint8)
    canvas[: pre.shape[0], : pre.shape[1]] = pre
    canvas[: post.shape[0], pre.shape[1] + gap :] = post
    return RefixResult(pre_op=pre, reduced=reduced, post_op=post, side_by_side=canvas, plan=plan, gap_vector=gv)


def save_refixation(image_bgr: np.ndarray, predicted_class: str, destination: Path, box=None) -> tuple[Path, HardwarePlan, Path]:
    result = simulate_refixation(image_bgr, predicted_class, box=box)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), result.post_op)
    side = destination.with_name(destination.stem + "_side_by_side.png")
    cv2.imwrite(str(side), result.side_by_side)
    return destination, result.plan, side
