# tools/prediction/routes.py
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import text
import os
import uuid
import traceback

from sqlalchemy import text
from extensions import db
from tools.prediction.services import run_prediction_pipeline

prediction_bp = Blueprint("prediction", __name__)

# =======================================================
# RUN PREDICTION
# =======================================================

# =====================================================================
# 🔵 RUN LTE PREDICTION PIPELINE
# =====================================================================
@prediction_bp.route("/run", methods=["POST"])
def run_prediction():
    """
    Input JSON:
    {
        "Project_id": 145,
        "Session_ids": [2685, 2683, 1690],
        "indoor_mode": "heuristic",
        "grid": 5
    }
    """

    payload = request.get_json()

    if not payload:
        return jsonify({"error": "No JSON body provided"}), 400

    # ---- Extract required fields ----
    project_id = payload.get("Project_id")
    session_ids = payload.get("Session_ids")
    indoor_mode = payload.get("indoor_mode", "heuristic")

    # validate numeric grid
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    project_id = data.get("Project_id")
    session_ids = data.get("Session_ids")
    indoor_mode = data.get("indoor_mode", "heuristic")

    try:
        pixel_size = float(payload.get("grid", 22.0))
        pixel_size = float(data.get("grid", 22.0))
    except:
        return jsonify({"error": "grid must be a valid number"}), 400
        pixel_size = 22.0

    # ---- Validate input ----
    if not project_id:
        return jsonify({"error": "Project_id is required"}), 400
    if not project_id or not session_ids:
        return jsonify({"error": "Project_id and Session_ids required"}), 400

    if not session_ids or not isinstance(session_ids, list):
        return jsonify({"error": "Session_ids must be a non-empty list"}), 400

    # ---- Setup output folder ----
    output_root = current_app.config.get(
        "OUTPUT_FOLDER",
        os.path.join(os.getcwd(), "outputs")
@@ -58,7 +38,6 @@ def run_prediction():
    run_id = str(uuid.uuid4())
    run_dir = os.path.join(output_root, f"lte_run_{run_id}")

    # ---- Execute pipeline ----
    try:
        with db.engine.begin() as conn:
            out_dir, count = run_prediction_pipeline(
@@ -71,55 +50,48 @@ def run_prediction():
            )

        return jsonify({
            "message": "Prediction completed successfully",
            "message": "Prediction successful",
            "project_id": project_id,
            "rows_saved": count,
            "output_folder": os.path.basename(out_dir),
            "grid_size_used": pixel_size,
            "rows_written": count,
            "output_dir": os.path.basename(out_dir),
            "run_id": run_id
        }), 200

    except Exception as e:
        current_app.logger.error(f"Prediction Error → {str(e)}")
        current_app.logger.error(f"Prediction Error: {str(e)}")
        current_app.logger.error(traceback.format_exc())

        return jsonify({
            "error": "Prediction pipeline failed",
            "detail": str(e)
        }), 500


# =====================================================================
# 🔵 DEBUG ENDPOINT
# Helps you check if the DB is prepared correctly for prediction.
# =====================================================================
# =======================================================
# DEBUG DB
# =======================================================

@prediction_bp.route("/debug-db/<int:project_id>", methods=["GET"])
def debug_database(project_id):
    try:
        out = {}

        results = {}
        with db.engine.connect() as conn:

            # list all tables
            tables = conn.execute(text("SHOW TABLES")).fetchall()
            out["all_tables"] = [t[0] for t in tables]
            results["tables"] = [t[0] for t in tables]

            # check project
            proj = conn.execute(
                text(f"SELECT * FROM tbl_project WHERE id = {project_id}")
                text(f"SELECT * FROM tbl_project WHERE id={project_id}")
            ).fetchone()
            out["project_exists"] = "YES" if proj else "NO"
            results["project_exists"] = bool(proj)

            # check site_noMl rows
            try:
                cnt = conn.execute(
                    text(f"SELECT COUNT(*) FROM site_noMl WHERE project_id={project_id}")
                ).scalar()
                out["site_noMl_count"] = cnt
                results["site_noMl_count"] = cnt
            except Exception as e:
                out["site_noMl_error"] = str(e)
                results["site_noMl_error"] = str(e)

        return jsonify(out), 200
        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
