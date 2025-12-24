# tools/prediction/routes.py

from flask import Blueprint, request, jsonify, current_app
import os
import uuid
import traceback
from sqlalchemy import text
from extensions import db
from tools.prediction.services import run_prediction_pipeline

prediction_bp = Blueprint("prediction", __name__)


# ============================================================
# 🔵 RUN PREDICTION PIPELINE
# ============================================================
@prediction_bp.route("/run", methods=["POST"])
def run_prediction():
    """
    Expects JSON:
    {
        "Project_id": 145,
        "Session_ids": [2685, 2683, 1690],
        "indoor_mode": "heuristic",
        "grid": 5
    }
    """

    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON body found"}), 400

    project_id = data.get("Project_id")
    session_ids = data.get("Session_ids")
    indoor_mode = data.get("indoor_mode", "heuristic")

    # Parse grid safely
    try:
        pixel_size = float(data.get("grid", 22.0))
        if pixel_size <= 0:
            raise ValueError
    except Exception:
        return jsonify({"error": "grid must be a positive numeric value"}), 400

    # Basic required fields
    if not project_id or not session_ids:
        return jsonify({
            "error": "Project_id and Session_ids are required"
        }), 400

    # Output directory
    output_base = current_app.config.get(
        "OUTPUT_FOLDER",
        os.path.join(os.getcwd(), "outputs")
    )
    run_id = str(uuid.uuid4())
    run_dir = os.path.join(output_base, f"lte_run_{run_id}")

    try:
        # Use DB transaction
        with db.engine.begin() as conn:

            out_dir, count = run_prediction_pipeline(
                db_connection=conn,
                project_id=str(project_id),
                session_ids=[str(s) for s in session_ids],
                outdir=run_dir,
                indoor_mode=indoor_mode,
                pixel_size_meters=pixel_size
            )

        return jsonify({
            "message": "Prediction successful",
            "project_id": project_id,
            "session_ids_used": session_ids,
            "grid_size": pixel_size,
            "predictions_saved": count,
            "output_directory": os.path.basename(out_dir),
            "run_id": run_id
        }), 200

    except Exception as e:
        current_app.logger.error(f"Prediction Error: {e}")
        current_app.logger.error(traceback.format_exc())

        return jsonify({
            "error": "Prediction pipeline failed",
            "detail": str(e)
        }), 500



# ============================================================
# 🔵 DEBUG DATABASE STATUS
# ============================================================
@prediction_bp.route("/debug-db/<int:project_id>", methods=["GET"])
def debug_database(project_id):

    try:
        results = {}

        with db.engine.connect() as conn:

            # 1. Check tables
            tables = conn.execute(text("SHOW TABLES")).fetchall()
            results["all_tables"] = [t[0] for t in tables]

            # 2. Check valid project
            proj = conn.execute(
                text(f"SELECT * FROM tbl_project WHERE id = {project_id}")
            ).fetchone()
            results["project_exists"] = "YES" if proj else "NO"

            # 3. Count site_noMl entries
            try:
                site_count = conn.execute(
                    text(f"SELECT COUNT(*) FROM site_noMl WHERE project_id = {project_id}")
                ).scalar()
                results["site_noMl_count"] = int(site_count)
            except Exception as e:
                results["site_noMl_error"] = str(e)

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
