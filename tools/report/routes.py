from flask import Blueprint, request, jsonify, current_app, send_file, redirect
import os
import threading
import uuid

from tools.report_engine.main import main as generate_report
from tools.report_engine.db import get_project_by_id
from extensions import db

report_bp = Blueprint("report", __name__)
REPORT_JOBS = {}
REPORT_JOBS_LOCK = threading.Lock()


def _safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def _set_job(report_id: str, **updates):
    with REPORT_JOBS_LOCK:
        state = REPORT_JOBS.get(report_id, {})
        state.update(updates)
        REPORT_JOBS[report_id] = state
        return dict(state)


def _get_job(report_id: str):
    with REPORT_JOBS_LOCK:
        state = REPORT_JOBS.get(report_id)
        return dict(state) if state else None


def background_report_task(app, project_id, user_id, report_id):
    with app.app_context():
        try:
            _set_job(report_id, status="processing")
            current_app.logger.info(
                f"[Report] Starting generation: project_id={project_id}, user_id={user_id}, report_id={report_id}"
            )
            generate_report(
                project_id=project_id,
                user_id=user_id,
                report_id=report_id,
                db_engine=db.engine,
            )
            project = get_project_by_id(project_id)
            download_url = (project or {}).get("Download_path")
            _set_job(
                report_id,
                status="ready",
                project_id=project_id,
                user_id=user_id,
                download_url=download_url,
            )
            current_app.logger.info(
                f"[Report] Completed generation: report_id={report_id}"
            )
        except Exception as e:
            _set_job(report_id, status="failed", error=str(e))
            current_app.logger.exception(
                f"[Report] Failed generation: report_id={report_id}"
            )


@report_bp.route("/generate", methods=["POST"])
def generate():
    data = request.get_json() or {}

    project_id = _safe_int(data.get("project_id") or data.get("Project_id"))
    user_id = _safe_int(data.get("user_id") or data.get("User_id"))

    if not project_id:
        return jsonify({"error": "project_id is required"}), 400
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    report_id = str(uuid.uuid4())
    _set_job(
        report_id,
        status="processing",
        project_id=project_id,
        user_id=user_id,
        download_url=None,
    )
    app = current_app._get_current_object()

    thread = threading.Thread(
        target=background_report_task,
        args=(app, project_id, user_id, report_id),
        daemon=True,
    )
    thread.start()

    return jsonify({
        "message": "Report generation started",
        "status": "processing",
        "project_id": project_id,
        "user_id": user_id,
        "report_id": report_id
    }), 202


@report_bp.route("/status/<report_id>", methods=["GET"])
def status(report_id):
    job = _get_job(report_id)
    if not job:
        return jsonify({
            "status": "not_found",
            "report_id": report_id,
        }), 404

    payload = {
        "status": job.get("status", "processing"),
        "report_id": report_id,
        "project_id": job.get("project_id"),
        "user_id": job.get("user_id"),
    }
    if job.get("download_url"):
        payload["download_url"] = job["download_url"]
    if job.get("error"):
        payload["error"] = job["error"]
    return jsonify(payload), 200


@report_bp.route("/download/<report_id>", methods=["GET"])
def download(report_id):
    reports_dir = os.path.join(current_app.root_path, "data", "reports")
    pdf_path = os.path.join(reports_dir, report_id, "report.pdf")

    if os.path.exists(pdf_path):
        return send_file(
            pdf_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="drive_test_report.pdf",
        )

    job = _get_job(report_id)
    if job and job.get("status") == "ready" and job.get("download_url"):
        return redirect(job["download_url"], code=302)

    return jsonify({"error": "Report not found"}), 404
