# /tools/prediction/routes.py
from flask import Blueprint, request, jsonify, current_app
import os
import uuid
import traceback
from sqlalchemy import text  # Required for raw SQL queries in debug endpoint
from extensions import db
from tools.prediction.services import run_prediction_pipeline

# Define the Blueprint
prediction_bp = Blueprint('prediction', __name__)

# ============================================================
# 🔵 RUN PREDICTION PIPELINE
# ============================================================
@prediction_bp.route('/run', methods=['POST'])
def run_prediction():
    """
    Endpoint to run the LTE Prediction Pipeline.
    Expects JSON:
    {
        "Project_id": 137,
        "Session_ids": [2747, 2745],
        "indoor_mode": "heuristic",
        "grid": 5
    }
    """
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
        
    project_id = data.get('Project_id')
    session_ids = data.get('Session_ids')
    indoor_mode = data.get('indoor_mode', 'heuristic')
    
    # Extract 'grid' from input (Default to 22.0 if missing or invalid)
    try:
        pixel_size = float(data.get('grid', 22.0))
    except (ValueError, TypeError):
        return jsonify({"error": "grid value must be a number"}), 400

    if not project_id or not session_ids:
        return jsonify({"error": "Missing Project_id or Session_ids"}), 400

    # Setup Output Directory
    output_base = current_app.config.get('OUTPUT_FOLDER', os.path.join(os.getcwd(), 'outputs'))
    run_id = str(uuid.uuid4())
    run_dir = os.path.join(output_base, f"lte_run_{run_id}")

    try:
        # Use a transaction connection
        with db.engine.begin() as connection:
            out_dir, count = run_prediction_pipeline(
                db_connection=connection,
                project_id=str(project_id),
                session_ids=[str(s) for s in session_ids],
                outdir=run_dir,
                indoor_mode=indoor_mode,
                pixel_size_meters=pixel_size
            )

        return jsonify({
            "message": "Prediction successful",
            "project_id": project_id,
            "predictions_saved": count,
            "output_directory": os.path.basename(out_dir),
            "grid_size_used": pixel_size,
            "run_id": run_id
        }), 200

    except Exception as e:
        current_app.logger.error(f"Prediction Error: {str(e)}")
        current_app.logger.error(traceback.format_exc())
        return jsonify({
            "error": "Prediction pipeline failed",
            "detail": str(e)
        }), 500


# ============================================================
# 🔵 DEBUG DATABASE (Add this to check your table status)
# ============================================================
# ============================================================
# 🔵 DEBUG DATABASE (Corrected Table Name)
# ============================================================
@prediction_bp.route('/debug-db/<int:project_id>', methods=['GET'])
def debug_database(project_id):
    try:
        results = {}
        with db.engine.connect() as conn:
            # 1. Check Tables
            tables = conn.execute(text("SHOW TABLES")).fetchall()
            results['all_tables'] = [t[0] for t in tables]
            
            # 2. Check Project (🟢 FIX: Use 'tbl_project')
            proj = conn.execute(text(f"SELECT * FROM tbl_project WHERE id = {project_id}")).fetchone()
            results['project_exists'] = "YES" if proj else "NO"
            
            # 3. Check Site Data (🟢 FIX: Use 'site_noMl')
            try:
                site_count = conn.execute(text(f"SELECT COUNT(*) FROM site_noMl WHERE project_id = {project_id}")).scalar()
                results['site_noMl_count'] = site_count
            except Exception as e:
                results['site_noMl_error'] = str(e)
                
        return jsonify(results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
