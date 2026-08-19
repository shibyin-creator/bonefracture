"""Flask clinical server with RBAC (admin vs clinician)."""

from __future__ import annotations

import json
import sys
import uuid
from functools import wraps
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import (
    Flask,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

from config import Config
from dataset_handler.catalog import load_catalog, probe_and_report
from models.pipeline import PIPELINE
from web_app.db import execute, fetchall, fetchone, init_db, log_audit, utcnow

app = Flask(
    __name__,
    template_folder=str(Config.WEB_DIR / "templates"),
    static_folder=str(Config.WEB_DIR / "static"),
)
app.config.from_object(Config)
Config.ensure_directories()
init_db()


def current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    row = fetchone("SELECT * FROM users WHERE id = ? AND is_active = 1", (user_id,))
    return dict(row) if row else None


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if user is None:
            if request.path.startswith("/api/"):
                return jsonify({"error": "authentication required"}), 401
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        user = current_user()
        if user is None or user["role"] != "admin":
            if request.path.startswith("/api/"):
                return jsonify({"error": "admin privileges required"}), 403
            flash("Administrator access required.", "error")
            return redirect(url_for("dashboard"))
        return view(*args, **kwargs)

    return wrapped


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


@app.context_processor
def inject_globals():
    return {
        "current_user": current_user(),
        "published_metrics": Config.PUBLISHED_METRICS,
        "morphology_classes": Config.MORPHOLOGY_CLASSES,
        "anatomical_classes": Config.ANATOMICAL_CLASSES,
    }


@app.route("/")
def index():
    if current_user():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user():
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        row = fetchone("SELECT * FROM users WHERE username = ?", (username,))
        if row and row["is_active"] and check_password_hash(row["password_hash"], password):
            session.clear()
            session.permanent = True
            session["user_id"] = row["id"]
            execute("UPDATE users SET last_login = ? WHERE id = ?", (utcnow(), row["id"]))
            log_audit(dict(row), "login", "Successful authentication", request.remote_addr or "")
            dest = request.args.get("next")
            if not dest:
                dest = url_for("admin_panel") if row["role"] == "admin" else url_for("dashboard")
            return redirect(dest)
        log_audit(None, "login_failed", f"username={username}", request.remote_addr or "")
        flash("Invalid credentials or inactive account.", "error")
    return render_template("login.html")


@app.route("/logout")
def logout():
    user = current_user()
    if user:
        log_audit(user, "logout", "", request.remote_addr or "")
    session.clear()
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    user = current_user()
    recent = fetchall(
        """
        SELECT * FROM inferences
        WHERE user_id = ? OR ? = 'admin'
        ORDER BY id DESC LIMIT 12
        """,
        (user["id"], user["role"]),
    )
    counts = fetchone("SELECT COUNT(*) AS n FROM inferences")
    severe = fetchone("SELECT COUNT(*) AS n FROM inferences WHERE needs_refixation = 1")
    return render_template(
        "dashboard.html",
        recent=[dict(r) for r in recent],
        total_studies=counts["n"] if counts else 0,
        severe_studies=severe["n"] if severe else 0,
        demo_mode=PIPELINE.demo_mode,
    )


@app.route("/clinical")
@login_required
def clinical_portal():
    return dashboard()


def _dataset_view_model(report: dict) -> dict:
    catalog = load_catalog()
    roles = {s["id"]: s for s in catalog["sources"]}
    remotes = []
    for source in catalog["sources"]:
        hit = next((r for r in report.get("remotes") or [] if r.get("id") == source["id"]), {})
        remotes.append(
            {
                "id": source["id"],
                "role": source.get("role"),
                "expected": source.get("expected_images"),
                "url": source.get("url") or source.get("kaggle_url"),
                "ok": hit.get("ok"),
                "detail": hit.get("detail"),
                "title": hit.get("title"),
            }
        )
    local = report.get("local") or {}
    graz = local.get("graz") or {}
    return {
        "remotes": remotes,
        "checklist": report.get("morphology_checklist") or [],
        "local": local,
        "graz_rows": graz.get("n_rows", 0),
        "graz_ok": bool(graz.get("match")),
        "generated_at": report.get("generated_at"),
        "roles": roles,
    }


@app.route("/datasets")
@login_required
def datasets_page():
    report_path = Config.DATASET_DIR / "inventory_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
    else:
        report = probe_and_report(probe=False)
    return render_template("datasets.html", **_dataset_view_model(report))


@app.route("/coverage")
@login_required
def coverage_page():
    atlas_path = ROOT / "docs" / "bone_coverage.json"
    atlas = json.loads(atlas_path.read_text())
    return render_template("coverage.html", atlas=atlas)


@app.route("/datasets/probe", methods=["POST"])
@login_required
def datasets_probe():
    try:
        report = probe_and_report(probe=True)
        flash("Live URLs probed. Inventory updated.", "ok")
    except Exception as exc:
        flash(f"Probe failed: {exc}", "error")
        report = probe_and_report(probe=False)
    log_audit(current_user(), "dataset_probe", "inventory refresh", request.remote_addr or "")
    return redirect(url_for("datasets_page"))


@app.route("/api/datasets")
@login_required
def api_datasets():
    report_path = Config.DATASET_DIR / "inventory_report.json"
    if report_path.exists():
        return jsonify(json.loads(report_path.read_text()))
    return jsonify(probe_and_report(probe=False))


@app.route("/refix/<int:study_id>")
@login_required
def refix_demo(study_id: int):
    row = fetchone("SELECT * FROM inferences WHERE id = ?", (study_id,))
    if row is None:
        flash("Study not found.", "error")
        return redirect(url_for("dashboard"))
    detections = json.loads(row["detections_json"] or "[]")
    return render_template("refix_demo.html", study=dict(row), detections=detections)


@app.route("/admin")
@admin_required
def admin_panel():
    users = [dict(r) for r in fetchall("SELECT id, username, full_name, role, is_active, created_at, last_login FROM users ORDER BY id")]
    logs = [dict(r) for r in fetchall("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 80")]
    studies = fetchone("SELECT COUNT(*) AS n FROM inferences")
    return render_template(
        "admin_panel.html",
        users=users,
        logs=logs,
        study_count=studies["n"] if studies else 0,
        weight_status={name: path.exists() for name, path in Config.WEIGHT_FILES.items()},
    )


@app.route("/api/admin/retrain", methods=["POST"])
@admin_required
def api_retrain():
    """Queue a documented retrain command; actual GPU training is started via main.py."""
    payload = request.get_json(silent=True) or {}
    model = payload.get("model", "softmax")
    log_audit(current_user(), "retrain_requested", f"model={model}", request.remote_addr or "")
    return jsonify(
        {
            "ok": True,
            "message": "Retrain acknowledged. Run on a GPU host:",
            "command": f"python main.py train --model {model}",
        }
    )


@app.route("/api/infer", methods=["POST"])
@login_required
def api_infer():
    user = current_user()
    upload = request.files.get("image") or request.files.get("file")
    if upload is None or not upload.filename:
        return jsonify({"error": "No radiograph uploaded."}), 400
    if not allowed_file(upload.filename):
        return jsonify({"error": "Unsupported file type. Use PNG, JPEG, TIFF, or DICOM."}), 400

    stem = f"{utcnow().replace(':', '').replace('-', '')}_{uuid.uuid4().hex[:8]}"
    safe_name = secure_filename(upload.filename)
    stored = f"{stem}_{safe_name}"
    dest = Config.UPLOAD_FOLDER / stored
    upload.save(dest)

    try:
        result = PIPELINE.analyze(dest, stem)
    except Exception as exc:
        log_audit(user, "inference_error", str(exc), request.remote_addr or "")
        return jsonify({"error": f"Inference failed: {exc}"}), 500

    study_id = execute(
        """
        INSERT INTO inferences (
            user_id, original_filename, stored_filename, detector_model, classifier_model,
            predicted_class, anatomical_site, confidence, detections_json, needs_refixation,
            gradcam_path, yolo_path, refix_path, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user["id"],
            safe_name,
            stored,
            result.detector_model,
            result.classifier_model,
            result.predicted_class,
            result.anatomical_site,
            result.confidence,
            json.dumps(result.to_dict()["detections"]),
            1 if result.needs_refixation else 0,
            result.gradcam_rel,
            result.yolo_rel,
            result.refix_rel,
            utcnow(),
        ),
    )
    log_audit(
        user,
        "inference",
        f"study={study_id} class={result.predicted_class} conf={result.confidence:.3f}",
        request.remote_addr or "",
    )
    payload = result.to_dict()
    payload["study_id"] = study_id
    payload["refix_url_page"] = url_for("refix_demo", study_id=study_id) if result.needs_refixation else None
    payload["side_by_side_url"] = result.side_by_side_rel
    return jsonify(payload)


@app.route("/api/metrics")
@login_required
def api_metrics():
    total = fetchone("SELECT COUNT(*) AS n FROM inferences")["n"]
    severe = fetchone("SELECT COUNT(*) AS n FROM inferences WHERE needs_refixation = 1")["n"]
    by_class = fetchall(
        """
        SELECT predicted_class AS label, COUNT(*) AS n
        FROM inferences
        WHERE predicted_class IS NOT NULL
        GROUP BY predicted_class
        ORDER BY n DESC
        """
    )
    return jsonify(
        {
            "published": Config.PUBLISHED_METRICS,
            "local": {
                "total_inferences": total,
                "severe_or_displaced": severe,
                "by_class": [dict(r) for r in by_class],
            },
            "classes": {
                "morphology": Config.MORPHOLOGY_CLASSES,
                "anatomical": Config.ANATOMICAL_CLASSES,
            },
            "weights": {name: path.exists() for name, path in Config.WEIGHT_FILES.items()},
        }
    )


@app.route("/api/users", methods=["GET", "POST"])
@admin_required
def api_users():
    if request.method == "GET":
        rows = fetchall(
            "SELECT id, username, full_name, role, is_active, created_at, last_login FROM users ORDER BY id"
        )
        return jsonify([dict(r) for r in rows])

    data = request.get_json(silent=True) or request.form
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    full_name = (data.get("full_name") or "").strip()
    role = data.get("role") or "clinician"
    if not username or not password:
        return jsonify({"error": "username and password are required"}), 400
    if role not in {"admin", "clinician", "radiologist"}:
        return jsonify({"error": "invalid role"}), 400
    try:
        user_id = execute(
            """
            INSERT INTO users (username, password_hash, full_name, role, is_active, created_at)
            VALUES (?, ?, ?, ?, 1, ?)
            """,
            (username, generate_password_hash(password), full_name, role, utcnow()),
        )
    except Exception:
        return jsonify({"error": "username already exists"}), 409
    log_audit(current_user(), "user_create", f"id={user_id} username={username}", request.remote_addr or "")
    return jsonify({"id": user_id, "username": username, "role": role}), 201


@app.route("/api/users/<int:user_id>", methods=["PATCH", "DELETE"])
@admin_required
def api_user_mutate(user_id: int):
    actor = current_user()
    target = fetchone("SELECT * FROM users WHERE id = ?", (user_id,))
    if target is None:
        return jsonify({"error": "user not found"}), 404

    if request.method == "DELETE":
        if target["username"] == Config.DEFAULT_ADMIN_USER:
            return jsonify({"error": "cannot delete the bootstrap administrator"}), 400
        execute("UPDATE users SET is_active = 0 WHERE id = ?", (user_id,))
        log_audit(actor, "user_deactivate", f"id={user_id}", request.remote_addr or "")
        return jsonify({"ok": True})

    data = request.get_json(silent=True) or {}
    fields = []
    params = []
    if "full_name" in data:
        fields.append("full_name = ?")
        params.append(data["full_name"])
    if "role" in data and data["role"] in {"admin", "clinician", "radiologist"}:
        fields.append("role = ?")
        params.append(data["role"])
    if "is_active" in data:
        fields.append("is_active = ?")
        params.append(1 if data["is_active"] else 0)
    if "password" in data and data["password"]:
        fields.append("password_hash = ?")
        params.append(generate_password_hash(data["password"]))
    if not fields:
        return jsonify({"error": "no updates"}), 400
    params.append(user_id)
    execute(f"UPDATE users SET {', '.join(fields)} WHERE id = ?", tuple(params))
    log_audit(actor, "user_update", f"id={user_id}", request.remote_addr or "")
    return jsonify({"ok": True})


@app.route("/api/audit")
@admin_required
def api_audit():
    rows = fetchall("SELECT * FROM audit_logs ORDER BY id DESC LIMIT 200")
    return jsonify([dict(r) for r in rows])


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "bone-fracture-system"})


@app.route("/static/outputs/<path:filename>")
@login_required
def outputs(filename: str):
    return send_from_directory(Config.OUTPUT_FOLDER, filename)


def main() -> None:
    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", 5000)), debug=False)


if __name__ == "__main__":
    main()
