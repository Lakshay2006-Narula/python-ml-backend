from flask import Blueprint, request, jsonify, current_app
import os
import uuid
import traceback
from extensions import db
from tools.prediction.services import run_prediction_pipeline

prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/run', methods=['POST'])
def run_prediction():
    """
    Endpoint to run the LTE Prediction Pipeline.
    Expects JSON:
    {
        "Project_id": 99,
        "Session_ids": [1629],
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
    
    # --- CHANGE: Extract 'grid' from input (Default to 5 if missing) ---
    try:
        # We look for "grid" in the input, but we use 22.0 as a backup
        pixel_size = float(data.get('grid', 22.0))
    except (ValueError, TypeError):
        return jsonify({"error": "grid value must be a number"}), 400
    # -------------------------------------------------------------------

    if not project_id or not session_ids:
        return jsonify({"error": "Missing Project_id or Session_ids"}), 400

    output_base = current_app.config.get('OUTPUT_FOLDER', os.path.join(os.getcwd(), 'outputs'))
    run_id = str(uuid.uuid4())
    run_dir = os.path.join(output_base, f"lte_run_{run_id}")

    try:
        with db.engine.begin() as connection:
            out_dir, count = run_prediction_pipeline(
                db_connection=connection,
                project_id=str(project_id),
                session_ids=[str(s) for s in session_ids],
                outdir=run_dir,
                indoor_mode=indoor_mode,
                # --- Pass the value to the function ---
                pixel_size_meters=pixel_size
                # --------------------------------------
            )

        return jsonify({
            "message": "Prediction successful",
            "project_id": project_id,
            "predictions_saved": count,
            "output_directory": out_dir,
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