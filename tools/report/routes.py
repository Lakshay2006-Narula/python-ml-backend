from flask import Blueprint, request, jsonify, current_app, send_file
import os
import threading
import uuid

from tools.report_engine.main import main as generate_report
from extensions import db

report_bp = Blueprint("report", __name__)


def _safe_int(value):
    try:
        return int(value)
    except Exception:
        return None


def background_report_task(app, project_id, user_id, report_id):
    with app.app_context():
        try:
            current_app.logger.info(
                f"[Report] Starting generation: project_id={project_id}, user_id={user_id}, report_id={report_id}"
            )
            generate_report(
                project_id=project_id,
                user_id=user_id,
                report_id=report_id,
                db_engine=db.engine,
            )
            current_app.logger.info(
                f"[Report] Completed generation: report_id={report_id}"
            )
        except Exception:
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


@report_bp.route("/download/<report_id>", methods=["GET"])
def download(report_id):
    reports_dir = os.path.join(current_app.root_path, "data", "reports")
    pdf_path = os.path.join(reports_dir, report_id, "report.pdf")

    if not os.path.exists(pdf_path):
        return jsonify({"error": "Report not found"}), 404

    return send_file(
        pdf_path,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="drive_test_report.pdf",
    )
