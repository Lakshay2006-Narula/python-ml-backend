from flask import Blueprint, request, jsonify, current_app
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

@prediction_bp.route("/run", methods=["POST"])
def run_prediction():
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    project_id = data.get("Project_id")
    session_ids = data.get("Session_ids")
    indoor_mode = data.get("indoor_mode", "heuristic")

    try:
        pixel_size = float(data.get("grid", 22.0))
    except:
        pixel_size = 22.0

    if not project_id or not session_ids:
        return jsonify({"error": "Project_id and Session_ids required"}), 400

    output_root = current_app.config.get(
        "OUTPUT_FOLDER",
        os.path.join(os.getcwd(), "outputs")
    )

    run_id = str(uuid.uuid4())
    run_dir = os.path.join(output_root, f"lte_run_{run_id}")

    try:
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
            "rows_written": count,
            "output_dir": os.path.basename(out_dir),
            "run_id": run_id
        }), 200

    except Exception as e:
        current_app.logger.error(f"Prediction Error: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "error": "Prediction pipeline failed",
            "detail": str(e)
        }), 500


# =======================================================
# DEBUG DB
# =======================================================

@prediction_bp.route("/debug-db/<int:project_id>", methods=["GET"])
def debug_database(project_id):
    try:
        results = {}
        with db.engine.connect() as conn:
            tables = conn.execute(text("SHOW TABLES")).fetchall()
            results["tables"] = [t[0] for t in tables]

            proj = conn.execute(
                text(f"SELECT * FROM tbl_project WHERE id={project_id}")
            ).fetchone()
            results["project_exists"] = bool(proj)

            try:
                cnt = conn.execute(
                    text(f"SELECT COUNT(*) FROM site_noMl WHERE project_id={project_id}")
                ).scalar()
                results["site_noMl_count"] = cnt
            except Exception as e:
                results["site_noMl_error"] = str(e)

        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
