#!/usr/bin/env python3
"""Build the 14-slide academic presentation for the bone-fracture project."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
OUT = ROOT / "Bone_Fracture_Detection_Presentation.pptx"

NAVY = RGBColor(7, 20, 28)
NAVY2 = RGBColor(11, 36, 48)
TEAL = RGBColor(46, 196, 182)
BLUE = RGBColor(61, 139, 253)
WHITE = RGBColor(244, 251, 252)
MUTED = RGBColor(176, 204, 214)
GOLD = RGBColor(244, 162, 97)
CARD = RGBColor(16, 42, 56)

W, H = Inches(13.333), Inches(7.5)


def _set_run(run, size=18, bold=False, color=WHITE, font="Calibri"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font


def add_text(box, text, size=18, bold=False, color=WHITE, align=PP_ALIGN.LEFT, font="Calibri"):
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    _set_run(run, size, bold, color, font)
    return tf


def add_para(tf, text, size=16, bold=False, color=WHITE, space_before=6, space_after=4, align=PP_ALIGN.LEFT):
    p = tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    run = p.add_run()
    run.text = text
    _set_run(run, size, bold, color)
    return p


def rect(slide, l, t, w, h, fill, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background() if line is None else None
    if line is not None:
        sh.line.color.rgb = line
        sh.line.width = Pt(1.25)
    else:
        sh.line.fill.background()
    # tighter corners
    try:
        sh.adjustments[0] = 0.08
    except Exception:
        pass
    return sh


def bg(slide, color=NAVY):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    return sh


def accent_bar(slide):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(0.12))
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()
    foot = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(7.22), W, Inches(0.28))
    foot.fill.solid()
    foot.fill.fore_color.rgb = RGBColor(5, 16, 22)
    foot.line.fill.background()
    tb = slide.shapes.add_textbox(Inches(0.4), Inches(7.24), Inches(9.5), Inches(0.24))
    add_text(
        tb,
        "Medical Image Analysis for Bone Fracture Detection Using Deep Learning  |  SHIBIN ANTONY  |  811241013",
        size=10,
        color=MUTED,
    )


def heading(slide, kicker, title):
    k = slide.shapes.add_textbox(Inches(0.5), Inches(0.28), Inches(12.2), Inches(0.28))
    add_text(k, kicker.upper(), size=12, bold=True, color=TEAL)
    t = slide.shapes.add_textbox(Inches(0.5), Inches(0.52), Inches(12.2), Inches(0.55))
    add_text(t, title, size=28, bold=True, color=WHITE)


def bullet_card(slide, l, t, w, h, title, lines, accent=TEAL):
    card = rect(slide, l, t, w, h, CARD, accent)
    ht = slide.shapes.add_textbox(l + Inches(0.18), t + Inches(0.12), w - Inches(0.3), Inches(0.36))
    add_text(ht, title, size=15, bold=True, color=accent)
    body = slide.shapes.add_textbox(l + Inches(0.18), t + Inches(0.46), w - Inches(0.32), h - Inches(0.55))
    tf = body.text_frame
    tf.clear()
    tf.word_wrap = True
    first = True
    for line in lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.space_after = Pt(6)
        run = p.add_run()
        run.text = "•  " + line
        _set_run(run, 13, False, WHITE)
    return card


def metric(slide, l, t, value, label, w=Inches(2.4)):
    rect(slide, l, t, w, Inches(1.15), CARD, TEAL)
    v = slide.shapes.add_textbox(l, t + Inches(0.12), w, Inches(0.55))
    add_text(v, value, size=26, bold=True, color=TEAL, align=PP_ALIGN.CENTER)
    lb = slide.shapes.add_textbox(l + Inches(0.08), t + Inches(0.62), w - Inches(0.16), Inches(0.45))
    add_text(lb, label, size=11, color=MUTED, align=PP_ALIGN.CENTER)


def build():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    blank = prs.slide_layouts[6]

    # --- 1 Title ---
    s = prs.slides.add_slide(blank)
    s.shapes.add_picture(str(ASSETS / "ppt_title_bg.png"), 0, 0, W, H)
    veil = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(8.4), H)
    veil.fill.solid()
    veil.fill.fore_color.rgb = RGBColor(6, 18, 26)
    veil.line.fill.background()
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.14), H)
    bar.fill.solid()
    bar.fill.fore_color.rgb = TEAL
    bar.line.fill.background()

    k = s.shapes.add_textbox(Inches(0.55), Inches(1.15), Inches(7.4), Inches(0.3))
    add_text(k, "MAJOR PROJECT  ·  ACADEMIC PRESENTATION", size=13, bold=True, color=TEAL)
    t = s.shapes.add_textbox(Inches(0.55), Inches(1.55), Inches(7.5), Inches(2.1))
    tf = add_text(t, "Medical Image Analysis for Bone Fracture Detection Using Deep Learning", size=30, bold=True, color=WHITE)
    add_para(tf, "YOLOv8  ·  VGG-16 Ensemble  ·  Grad-CAM  ·  Surgical Refixation Simulation", size=14, color=MUTED, space_before=10)

    rect(s, Inches(0.55), Inches(4.15), Inches(7.2), Inches(2.35), CARD, TEAL)
    info = s.shapes.add_textbox(Inches(0.75), Inches(4.28), Inches(6.8), Inches(2.1))
    tf = add_text(info, "Submitted by", size=11, bold=True, color=TEAL)
    add_para(tf, "SHIBIN ANTONY", size=22, bold=True, color=WHITE, space_before=4)
    add_para(tf, "Enrollment No.  811241013", size=16, color=MUTED, space_before=2)
    add_para(tf, "Under the Guidance of", size=11, bold=True, color=TEAL, space_before=12)
    add_para(tf, "Project Guide", size=18, bold=True, color=WHITE, space_before=2)
    add_para(tf, "(Insert faculty name and designation)", size=11, color=MUTED, space_before=0)

    # --- 2 Abstract ---
    s = prs.slides.add_slide(blank)
    bg(s)
    accent_bar(s)
    heading(s, "Slide 02  ·  Overview", "Abstract")
    rect(s, Inches(0.5), Inches(1.25), Inches(12.3), Inches(5.7), CARD)
    ab = s.shapes.add_textbox(Inches(0.75), Inches(1.45), Inches(11.8), Inches(5.35))
    tf = add_text(
        ab,
        "Bone fractures are among the most common musculoskeletal injuries and require timely, accurate diagnosis for effective treatment. This project presents a web-based clinical workstation that assists medical professionals in detecting and classifying fractures from X-ray and DICOM images with high precision.",
        size=16,
        color=WHITE,
    )
    add_para(
        tf,
        "The core detector is YOLOv8, trained and validated on 3,316 training and 399 validation images covering anatomical sites such as elbow, fingers, forearm, humerus, shoulder, and wrist, achieving a mean Average Precision (mAP50) of 86%. Fracture morphology is classified using the IEEE Access 2025 stack: VGG-16 with Softmax, VGG-16 with Random Forest (both 95% accuracy), ResNet-50 with SVM (93%), and EfficientNetB0 with XGBoost (comparative baseline).",
        size=16,
        color=WHITE,
        space_before=12,
    )
    add_para(
        tf,
        "Explainable AI (Grad-CAM) highlights suspected fracture regions, while severe or displaced patterns automatically generate an educational post-surgical refixation overlay (plates, screws, or intramedullary rods). A Flask dashboard with secure admin login, audit logs, and dark/light clinical UI makes the system usable in local, Jupyter, and Google Colab environments—supporting screening where radiology expertise is limited.",
        size=16,
        color=WHITE,
        space_before=12,
    )

    # --- 3 Contents ---
    s = prs.slides.add_slide(blank)
    bg(s)
    accent_bar(s)
    heading(s, "Slide 03  ·  Agenda", "Contents")
    items = [
        ("01", "Title"),
        ("02", "Abstract"),
        ("03", "Contents"),
        ("04", "Introduction"),
        ("05", "Problem Statement"),
        ("06", "Objectives"),
        ("07", "Existing System"),
        ("08", "Proposed System"),
        ("09", "Software Requirements"),
        ("10", "Hardware Requirements"),
        ("11", "System Architecture"),
        ("12", "Dataset"),
        ("13", "Algorithms Used"),
        ("14", "Conclusion"),
    ]
    for i, (num, name) in enumerate(items):
        col = i // 7
        row = i % 7
        l = Inches(0.55 + col * 6.3)
        t = Inches(1.25 + row * 0.78)
        rect(s, l, t, Inches(5.95), Inches(0.68), CARD, TEAL if i % 2 == 0 else BLUE)
        n = s.shapes.add_textbox(l + Inches(0.18), t + Inches(0.14), Inches(0.7), Inches(0.4))
        add_text(n, num, size=18, bold=True, color=TEAL)
        nm = s.shapes.add_textbox(l + Inches(0.95), t + Inches(0.16), Inches(4.7), Inches(0.4))
        add_text(nm, name, size=18, color=WHITE)

    # --- 4 Introduction ---
    s = prs.slides.add_slide(blank)
    bg(s)
    accent_bar(s)
    heading(s, "Slide 04  ·  Context", "Introduction")
    intro_cards = [
        (
            "Clinical burden",
            [
                "Millions of fractures occur yearly from trauma, sports, falls, and osteoporosis.",
                "Correct type and site decide conservative care versus operative fixation.",
                "Delay or misclassification can lead to malunion, non-union, or joint damage.",
            ],
        ),
        (
            "Diagnostic gap",
            [
                "Manual radiograph reading is time-consuming and observer-dependent.",
                "Hairline, greenstick, and overlapping anatomy are easy to miss.",
                "Many regions lack round-the-clock specialist radiology support.",
            ],
        ),
        (
            "Role of deep learning",
            [
                "CNNs extract subtle cortical disruption that is hard to verbalize.",
                "Object detection localizes the injury; classifiers name the pattern.",
                "XAI heatmaps keep the clinician in the loop instead of a black box.",
            ],
        ),
    ]
    for i, (title, lines) in enumerate(intro_cards):
        bullet_card(s, Inches(0.45 + i * 4.2), Inches(1.3), Inches(4.0), Inches(5.55), title, lines)

    # --- 5 Problem statement ---
    s = prs.slides.add_slide(blank)
    bg(s)
    accent_bar(s)
    heading(s, "Slide 05  ·  Need", "Problem Statement")
    problems = [
        ("01", "Subjectivity", "Inter-observer variability: the same X-ray may be labelled differently by two readers, especially for hairline and pathological fractures."),
        ("02", "Latency", "Emergency and trauma workflows need seconds-level triage, not a long queue for specialist review."),
        ("03", "Narrow tools", "Many published systems handle binary “fracture vs normal” or a few limb sites, not 10+ morphology classes plus full-body anatomy."),
        ("04", "No action layer", "Detection alone does not communicate likely fixation strategy (plate, rod, screws) for severe displaced injuries."),
        ("05", "Weak explainability", "Pure accuracy scores do not show where the model looked, reducing clinical trust."),
        ("06", "Deployment gap", "Research notebooks rarely become a secure, auditable hospital-style web workstation."),
    ]
    for i, (num, title, body) in enumerate(problems):
        col, row = i % 3, i // 3
        l, t = Inches(0.45 + col * 4.2), Inches(1.28 + row * 2.85)
        rect(s, l, t, Inches(4.0), Inches(2.65), CARD, GOLD if i in (0, 3) else TEAL)
        n = s.shapes.add_textbox(l + Inches(0.18), t + Inches(0.16), Inches(3.6), Inches(0.32))
        add_text(n, f"{num}   {title}", size=16, bold=True, color=TEAL)
        b = s.shapes.add_textbox(l + Inches(0.18), t + Inches(0.55), Inches(3.65), Inches(1.9))
        add_text(b, body, size=13, color=WHITE)

    # --- 6 Objectives ---
    s = prs.slides.add_slide(blank)
    bg(s)
    accent_bar(s)
    heading(s, "Slide 06  ·  Goals", "Objectives")
    objs = [
        "Design a medical-grade Flask workstation for real-time X-ray / DICOM fracture analysis.",
        "Localize fractures with YOLOv8 bounding boxes across anatomical sites (upper limb, shoulder, pelvis).",
        "Classify 10 morphological types using VGG-16 Softmax and VGG-16 + Random Forest as primary models.",
        "Benchmark ResNet-50 + SVM and EfficientNetB0 + XGBoost as comparative ensembles (IEEE Access 2025).",
        "Integrate Grad-CAM heatmaps so clinicians can verify the region driving the prediction.",
        "Auto-generate an educational post-surgical refixation overlay for severe / displaced patterns.",
        "Provide secure admin authentication, user roles, metric monitoring, and audit logging.",
        "Support training and inference on local machines, Jupyter, and Google Colab at 20,000+ image scale.",
    ]
    for i, line in enumerate(objs):
        t = Inches(1.22 + i * 0.70)
        rect(s, Inches(0.5), t, Inches(12.3), Inches(0.62), CARD, TEAL)
        n = s.shapes.add_textbox(Inches(0.65), t + Inches(0.12), Inches(0.55), Inches(0.4))
        add_text(n, f"{i+1:02d}", size=16, bold=True, color=TEAL)
        b = s.shapes.add_textbox(Inches(1.25), t + Inches(0.12), Inches(11.3), Inches(0.42))
        add_text(b, line, size=15, color=WHITE)

    # --- 7 Existing system ---
    s = prs.slides.add_slide(blank)
    bg(s)
    accent_bar(s)
    heading(s, "Slide 07  ·  Prior art", "Existing System")
    bullet_card(
        s,
        Inches(0.45),
        Inches(1.25),
        Inches(6.1),
        Inches(5.6),
        "Conventional practice",
        [
            "Radiologists inspect radiographs manually using experience and side markers.",
            "Reports depend on lighting, fatigue, and case mix; subtle fractures are missed.",
            "Earlier CAD tools were often binary (fracture / no fracture) on a single bone.",
            "Transfer-learning papers used VGG-16 or ResNet-50 in isolation, without a clinical UI.",
            "Few systems couple detection, multi-class typing, XAI, and fixation education.",
        ],
        BLUE,
    )
    bullet_card(
        s,
        Inches(6.75),
        Inches(1.25),
        Inches(6.1),
        Inches(5.6),
        "Limitations observed",
        [
            "IEEE Access 2025: EfficientNetB0 + XGBoost lagged (~41% accuracy) on grayscale X-rays.",
            "Small, imbalanced public sets under-represent longitudinal and oblique fractures.",
            "No standard pathway from a notebook metric to an auditable hospital dashboard.",
            "Absence of surgical-context visualization for comminuted, spiral, and dislocation injuries.",
            "Limited DICOM ingest and no role-based access in typical academic demos.",
        ],
        GOLD,
    )

    # --- 8 Proposed system ---
    s = prs.slides.add_slide(blank)
    bg(s)
    accent_bar(s)
    heading(s, "Slide 08  ·  Solution", "Proposed System")
    props = [
        ("FractureNet workstation", "Flask + HTML/CSS/JS clinical UI with dark/light mode, canvas viewer, and admin governance."),
        ("Dual-head AI", "YOLOv8 finds where the injury is; VGG-16 ensembles decide what type it is."),
        ("Explainability", "Grad-CAM overlays the last convolutional evidence so predictions are visually inspectable."),
        ("Refixation demo", "Severe classes trigger plate / IM-rod / screw simulation via OpenCV blending and an interactive canvas."),
        ("Comparative science", "Same data protocol as the paper: VGG-16, VGG-16+RF, ResNet-50+SVM, EfficientNetB0+XGBoost."),
        ("Ops-ready", "SQLite users, hashed passwords, inference history, audit trail; Colab notebook for large-scale training."),
    ]
    for i, (title, body) in enumerate(props):
        col, row = i % 3, i // 3
        l, t = Inches(0.45 + col * 4.2), Inches(1.28 + row * 2.85)
        rect(s, l, t, Inches(4.0), Inches(2.65), CARD, TEAL)
        ht = s.shapes.add_textbox(l + Inches(0.2), t + Inches(0.2), Inches(3.6), Inches(0.7))
        add_text(ht, title, size=16, bold=True, color=TEAL)
        bd = s.shapes.add_textbox(l + Inches(0.2), t + Inches(0.95), Inches(3.6), Inches(1.45))
        add_text(bd, body, size=14, color=WHITE)

    # --- 9 Software ---
    s = prs.slides.add_slide(blank)
    bg(s)
    accent_bar(s)
    heading(s, "Slide 09  ·  Stack", "Software Requirements")
    sw = [
        ("Language & web", ["Python 3.10+", "Flask, Jinja2, Werkzeug", "HTML5, CSS3, JavaScript", "SQLite (users, audits, studies)"]),
        ("Deep learning", ["PyTorch + Ultralytics YOLOv8", "TensorFlow / Keras VGG-16, ResNet-50, EfficientNetB0", "CUDA GPU builds when available"]),
        ("Classical ML & imaging", ["Scikit-learn (Random Forest, SVM)", "XGBoost", "OpenCV, Pillow, pydicom", "NumPy, Pandas, Matplotlib"]),
        ("Training & ops", ["Jupyter / Google Colab notebook", "Joblib model serialization", "TensorBoard + ModelCheckpoint", "Git version control"]),
    ]
    for i, (title, lines) in enumerate(sw):
        col, row = i % 2, i // 2
        bullet_card(s, Inches(0.45 + col * 6.4), Inches(1.25 + row * 2.85), Inches(6.15), Inches(2.7), title, lines)

    # --- 10 Hardware ---
    s = prs.slides.add_slide(blank)
    bg(s)
    accent_bar(s)
    heading(s, "Slide 10  ·  Infrastructure", "Hardware Requirements")
    hw = [
        ("Minimum (demo / CPU)", "4-core CPU, 8 GB RAM, 20 GB disk. Runs the Flask UI and OpenCV demo pipeline without trained heavyweights."),
        ("Recommended training", "NVIDIA GPU (T4 / RTX 8 GB+), 16–32 GB RAM, 50+ GB SSD for 20k-scale datasets, checkpoints, and YOLO runs."),
        ("Colab / cloud", "Google Colab GPU runtime or equivalent cloud VM; optional Google Drive for dataset and weight export."),
        ("Clinical workstation", "Full-HD display for side-by-side original / YOLO / Grad-CAM / refixation; optional DICOM-capable storage."),
        ("Network", "Localhost for on-prem demo; HTTPS reverse proxy if exposed on a hospital LAN (change default secrets)."),
        ("Peripherals", "Keyboard/mouse for canvas hardware placement; scanner or PACS export for radiograph ingest."),
    ]
    for i, (title, body) in enumerate(hw):
        col, row = i % 3, i // 3
        l, t = Inches(0.45 + col * 4.2), Inches(1.28 + row * 2.85)
        rect(s, l, t, Inches(4.0), Inches(2.65), CARD, BLUE)
        ht = s.shapes.add_textbox(l + Inches(0.18), t + Inches(0.18), Inches(3.65), Inches(0.7))
        add_text(ht, title, size=15, bold=True, color=TEAL)
        bd = s.shapes.add_textbox(l + Inches(0.18), t + Inches(0.95), Inches(3.65), Inches(1.5))
        add_text(bd, body, size=13, color=WHITE)

    # --- 11 Architecture ---
    s = prs.slides.add_slide(blank)
    bg(s)
    accent_bar(s)
    heading(s, "Slide 11  ·  Design", "System Architecture")
    s.shapes.add_picture(str(ASSETS / "ppt_architecture.png"), Inches(0.4), Inches(1.15), Inches(8.35), Inches(4.7))
    rect(s, Inches(8.9), Inches(1.2), Inches(3.95), Inches(5.6), CARD, TEAL)
    side = s.shapes.add_textbox(Inches(9.1), Inches(1.35), Inches(3.55), Inches(5.3))
    tf = add_text(side, "Pipeline", size=16, bold=True, color=TEAL)
    for line in [
        "1. Authenticated upload (PNG/JPEG/DICOM)",
        "2. Preprocess & CLAHE enhancement",
        "3. YOLOv8 localization",
        "4. VGG-16 / RF classification",
        "5. Comparative heads (optional)",
        "6. Grad-CAM explanation",
        "7. Severity gate → refixation overlay",
        "8. Persist study + audit log",
        "9. Clinical dashboard review",
    ]:
        add_para(tf, line, size=12, color=WHITE, space_before=8)

    # --- 12 Dataset ---
    s = prs.slides.add_slide(blank)
    bg(s)
    accent_bar(s)
    heading(s, "Slide 12  ·  Data", "Dataset")
    metric(s, Inches(0.45), Inches(1.22), "10", "Morphology classes\n(IEEE Access 2025)")
    metric(s, Inches(3.05), Inches(1.22), "8", "Anatomical sites\n(+ pelvis)")
    metric(s, Inches(5.65), Inches(1.22), "3,316 / 399", "YOLO train / val\n(abstract protocol)")
    metric(s, Inches(8.25), Inches(1.22), "20k+", "Target scale\n(Colab / multi-source)")
    metric(s, Inches(10.85), Inches(1.22), "256²", "Classifier input\nRGB X-rays")

    morph = "Avulsion · Comminuted · Fracture-Dislocation · Greenstick · Hairline · Impacted · Longitudinal · Oblique · Pathological · Spiral"
    anat = "Elbow Positive · Fingers Positive · Forearm Fracture · Humerus · Humerus Fracture · Shoulder Fracture · Wrist Positive · Pelvic Fracture"
    rect(s, Inches(0.45), Inches(2.6), Inches(12.4), Inches(1.55), CARD, TEAL)
    b = s.shapes.add_textbox(Inches(0.65), Inches(2.72), Inches(12.0), Inches(1.35))
    tf = add_text(b, "Morphological classes", size=13, bold=True, color=TEAL)
    add_para(tf, morph, size=14, color=WHITE, space_before=6)

    rect(s, Inches(0.45), Inches(4.3), Inches(12.4), Inches(1.55), CARD, BLUE)
    b = s.shapes.add_textbox(Inches(0.65), Inches(4.42), Inches(12.0), Inches(1.35))
    tf = add_text(b, "Anatomical classes", size=13, bold=True, color=BLUE)
    add_para(tf, anat, size=14, color=WHITE, space_before=6)

    note = s.shapes.add_textbox(Inches(0.5), Inches(6.0), Inches(12.3), Inches(1.0))
    add_text(
        note,
        "Paper set: 1,129 labelled X-rays (989 train / 140 test, 80/20 val split), grayscale converted to 256×256×3. YOLO set uses Ultralytics labels. Public Kaggle fracture collections can be merged to reach 20,000+ images; class imbalance (longitudinal / oblique under-represented) is handled with augmentation and balanced forests.",
        size=13,
        color=MUTED,
    )

    # --- 13 Algorithms ---
    s = prs.slides.add_slide(blank)
    bg(s)
    accent_bar(s)
    heading(s, "Slide 13  ·  Methods", "Algorithms Used")
    algos = [
        ("YOLOv8", "Primary detector. Single-pass localization and anatomical labels. Abstract mAP50 = 86%."),
        ("VGG-16 + Softmax", "Frozen ImageNet backbone, Dense–ReLU–Dropout–Softmax, Adam 5e-4, CCE, 20 epochs. Accuracy 0.95."),
        ("VGG-16 + Random Forest", "Avg-pooled VGG features → 300-tree balanced RF. Top hybrid in the reference paper (0.95)."),
        ("ResNet-50 + SVM", "Residual features + linear SVM. Comparative ensemble, ~0.93 accuracy."),
        ("EfficientNetB0 + XGBoost", "Compound-scaled CNN features + gradient boosting. Paper baseline (~0.41) kept for fair comparison."),
        ("Grad-CAM + Refixation", "Last-conv heatmaps for XAI. AO-style plate / IM nail / screw overlay for severe classes."),
    ]
    for i, (title, body) in enumerate(algos):
        col, row = i % 3, i // 3
        l, t = Inches(0.45 + col * 4.2), Inches(1.25 + row * 2.85)
        rect(s, l, t, Inches(4.0), Inches(2.65), CARD, TEAL)
        ht = s.shapes.add_textbox(l + Inches(0.18), t + Inches(0.18), Inches(3.65), Inches(0.55))
        add_text(ht, title, size=16, bold=True, color=TEAL)
        bd = s.shapes.add_textbox(l + Inches(0.18), t + Inches(0.8), Inches(3.65), Inches(1.6))
        add_text(bd, body, size=13, color=WHITE)

    # --- 14 Conclusion ---
    s = prs.slides.add_slide(blank)
    bg(s)
    accent_bar(s)
    heading(s, "Slide 14  ·  Closing", "Conclusion")
    metric(s, Inches(0.5), Inches(1.25), "95%", "VGG-16 / VGG-16+RF\nclassification accuracy")
    metric(s, Inches(3.7), Inches(1.25), "86%", "YOLOv8 mAP50\nreal-time detection")
    metric(s, Inches(6.9), Inches(1.25), "93%", "ResNet-50 + SVM\ncomparative hybrid")
    metric(s, Inches(10.1), Inches(1.25), "18", "Unified classes\nmorphology + anatomy")

    rect(s, Inches(0.5), Inches(2.7), Inches(12.3), Inches(4.15), CARD, TEAL)
    c = s.shapes.add_textbox(Inches(0.75), Inches(2.9), Inches(11.8), Inches(3.85))
    tf = add_text(
        c,
        "This work delivers an end-to-end, IEEE-aligned platform: YOLOv8 finds the fracture, VGG-16 ensembles name it, Grad-CAM explains it, and a refixation module illustrates likely hardware for severe injuries—inside a secure clinical web app.",
        size=16,
        color=WHITE,
    )
    add_para(
        tf,
        "Results align with published evidence that VGG-16-based models outperform lighter hybrids on this X-ray taxonomy. The system is intended as decision support and education—not a certified medical device—and every output requires licensed clinical review.",
        size=15,
        color=WHITE,
        space_before=10,
    )
    add_para(tf, "Future work", size=15, bold=True, color=TEAL, space_before=12)
    add_para(
        tf,
        "• Multi-centre 20k+ labelled set  ·  Unfreeze later VGG blocks  ·  3D CT/MRI fusion  ·  Prospective reader study  ·  PACS/HL7 integration  ·  Regulatory-grade validation",
        size=14,
        color=WHITE,
        space_before=4,
    )
    add_para(tf, "Thank you  ·  Questions welcome", size=18, bold=True, color=TEAL, space_before=14, align=PP_ALIGN.CENTER)

    prs.save(OUT)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
