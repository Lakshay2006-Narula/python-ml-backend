# In tools/cell_site/routes.py

from flask import Blueprint, request, jsonify, send_file, current_app
import os
import traceback
import json 
from models import SiteNoMl
import re   # <- you have this already
import io   # <- ADD THIS
import csv 

# === THIS IS THE CORRECT IMPORT ORDER ===
# 1. Import from extensions and models first
from extensions import db
from models import Prediction 

# 2. THEN import your service
from .services import CellSiteService
# === END OF FIX ===

# Setup Blueprint
cell_site_bp = Blueprint('cell_site', __name__)

# ---
# --- 🔴 BUG WAS HERE 🔴 ---
# We REMOVED this line: service = CellSiteService()
# Creating the service here makes it "global" and shared by all requests,
# which causes the "already exists" error.
# ---

@cell_site_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'tool': 'Cell Site Locator',
        'version': '1.0.0',
        'endpoints': ['/upload', '/download/<output_dir>/<filename>']
    })


# ============================================
# 🔧 FIXED: Added OPTIONS method
# ============================================
@cell_site_bp.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    """Upload and process cell site data"""
    
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 204
    
    try:
        # ---
        # --- 🟢 SOLUTION IS HERE 🟢 ---
        # We create a NEW service object for EVERY request.
        # This ensures there is no "stale" or "cached" data from
        # previous requests.
        service = CellSiteService()
        # ---

        # --- 1. File Validation ---
        if 'file' not in request.files:
            current_app.logger.error("No file in request.files")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        
        if file.filename == '':
            current_app.logger.error("Empty filename")
            return jsonify({'error': 'No file selected'}), 400

        # Now this line uses the NEW service object
        if not service.allowed_file(file.filename):
            current_app.logger.error(f"Invalid file type: {file.filename}")
            return jsonify({
                'error': 'Invalid file type',
                'allowed': list(service.ALLOWED_EXTENSIONS)
            }), 400

        # --- 2. Extract Form Parameters ---
        params = {
            'method': request.form.get('method', 'noml'),
            'min_samples': int(request.form.get('min_samples', 30)),
            'bin_size': int(request.form.get('bin_size', 5)),
            'soft_spacing': request.form.get('soft_spacing', 'false').lower() == 'true',
            'use_ta': request.form.get('use_ta', 'false').lower() == 'true',
            'make_map': request.form.get('make_map', 'false').lower() == 'true',
            'model_path': request.form.get('model_path'),
            'train_path': request.form.get('train_path')
        }

        # --- 3. Extract Project ID ---
        project_id = None
        if 'project_id' in request.form:
            # We get an integer
            project_id = request.form.get('project_id', type=int)
            current_app.logger.info(f"Got project_id from form: {project_id}")
        elif 'project_data' in request.form:
            try:
                data = json.loads(request.form.get('project_data'))
                project_id = data.get('Project_Id') # Match your JSON
                current_app.logger.info(f"Got project_id from project_data: {project_id}")
            except (json.JSONDecodeError, TypeError) as e:
                current_app.logger.warning(f"Invalid JSON in 'project_data' field: {e}")
                pass 

        current_app.logger.info(f"Processing file: {file.filename} with method: {params['method']} for Project: {project_id}")

        # --- 4. Call the Service ---
        # This line also uses the NEW service object
        result = service.process_file(file, params, project_id)

        if not result:
            current_app.logger.error("Service returned empty result")
            return jsonify({'error': 'Processing failed - no result returned'}), 500

        # --- 5. Log to 'predictions' Table ---
        if result and result.get('output_dir') and result.get('results'):
            results_dict = result.get('results') 

            if results_dict and isinstance(results_dict, dict) and results_dict.values():
                 output_filename = list(results_dict.values())[0]
            else:
                 output_filename = 'no_file_generated'

            # --- 🟢 CORRECTED SYNTAX HERE ---
            # Changed 'key': value to key=value
            new_prediction = Prediction(
                output_dir=result['output_dir'],
                filename=output_filename,
                method=params['method'],
                min_samples=params['min_samples'],
                project_id=project_id
            )
            # --- END OF FIX ---

            db.session.add(new_prediction)
            db.session.commit()

            current_app.logger.info(f"Saved prediction {new_prediction.id} to log database.")

        return jsonify(result), 200

    except ValueError as e:
        current_app.logger.error(f"Validation error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': f'Validation error: {str(e)}', 'type': 'ValueError'}), 400
        
    except Exception as e:
        db.session.rollback() 
        current_app.logger.error(f"Upload error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500
    
from werkzeug.datastructures import FileStorage

def safe_int(value):
    try:
        return int(float(value))
    except:
        return None

def safe_float(value):
    try:
        return float(value)
    except:
        return None

def extract_mci(cell_info_str):
    if not cell_info_str:
        return None
    match = re.search(r'mCi=([0-9*]+)', cell_info_str)
    return match.group(1) if match else None


@cell_site_bp.route('/process-session', methods=['POST'])
def process_session():
    from models import NetworkLog
    data = request.get_json()

    session_ids = data.get('session_ids')
    project_id = data.get('project_id')

    if not session_ids or not isinstance(session_ids, list):
        return jsonify({"error": "session_ids must be a list"}), 400

    if not project_id:
        return jsonify({"error": "project_id is required"}), 400

    logs = NetworkLog.query.filter(NetworkLog.session_id.in_(session_ids)).all()
    if not logs:
        return jsonify({"error": "No logs found for given session_ids"}), 404

    # Create CSV in memory
    csv_buffer = io.BytesIO()
    writer = csv.writer(io.TextIOWrapper(csv_buffer, encoding='utf-8', newline=''))

    writer.writerow([
        'timestamp_utc', 'lat', 'lon', 'network', 'technology',
        'earfcn_or_narfcn', 'pci_or_psi', 'rsrp_dbm', 'rsrq_db',
        'sinr_db', 'band_mhz', 'cell_id_global', 'ta'
    ])

    for log in logs:
        writer.writerow([
            log.timestamp.isoformat() if log.timestamp else None,
            safe_float(log.lat),
            safe_float(log.lon),
            log.m_alpha_long,     # network cluster label
            log.network,          # technology LTE/NR
            safe_int(log.earfcn),
            safe_int(log.pci),
            safe_float(log.rsrp),
            safe_float(log.rsrq),
            safe_float(log.sinr),
            log.band,
            extract_mci(log.primary_cell_info_1),
            log.ta
        ])

    # ✅ Convert BytesIO -> FileStorage so process_file() can .save()
    csv_buffer.seek(0)
    file_like = FileStorage(
        stream=csv_buffer,
        filename=f"project_{project_id}.csv",
        content_type="text/csv"
    )

    service = CellSiteService()

    params = {
        'method': 'noml',     # (we will upgrade this to auto mode later)
        'min_samples': 30,
        'bin_size': 5,
        'soft_spacing': False,
        'use_ta': False,
        'make_map': True
    }

    result = service.process_file(file_like, params, project_id)

    if not result:
        return jsonify({'error': 'Processing failed'}), 500
    
    # ✅ Log into predictions table
    from models import Prediction
    from extensions import db
    
    if result and result.get('output_dir') and result.get('results'):
        results_dict = result.get('results')
    
        if results_dict and isinstance(results_dict, dict) and results_dict.values():
            output_filename = list(results_dict.values())[0]
        else:
            output_filename = 'no_file_generated'
    
        new_prediction = Prediction(
            output_dir=result['output_dir'],
            filename=output_filename,
            method=params['method'],
            min_samples=params['min_samples'],
            project_id=project_id
        )
    
        db.session.add(new_prediction)
        db.session.commit()


    return jsonify({
        "status": "success",
        "project_id": project_id,
        "result": result
    }), 200




@cell_site_bp.route('/download/<output_dir>/<filename>', methods=['GET'])
def download_file(output_dir, filename):
    """Download generated files"""
    try:
        safe_output_dir = os.path.basename(output_dir)
        safe_filename = os.path.basename(filename)
        
        file_path = os.path.join(
            current_app.config['OUTPUT_FOLDER'],
            safe_output_dir,
            safe_filename
        )
        
        if not os.path.exists(file_path):
            current_app.logger.error(f"File not found: {file_path}")
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True, download_name=safe_filename)
        
    except Exception as e:
        current_app.logger.error(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@cell_site_bp.route('/outputs/<output_dir>', methods=['GET'])
def list_outputs(output_dir):
    """List all files in an output directory"""
    try:
        safe_output_dir = os.path.basename(output_dir)
        
        dir_path = os.path.join(current_app.config['OUTPUT_FOLDER'], safe_output_dir)
        
        if not os.path.exists(dir_path) or not os.path.isdir(dir_path):
            return jsonify({'error': 'Directory not found'}), 404
            
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        
        return jsonify({'files': files, 'count': len(files)}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================
# 🔧 FIXED: Added OPTIONS method
# ============================================
@cell_site_bp.route('/update-project-id', methods=['POST', 'OPTIONS'])
def update_prediction_project_id():
    """
    Updates the project_id for an existing prediction log entry,
    based on the filename.
    
    Expects JSON:
    {
        "filename": "some_file.csv",
        "Project_Id": 12345
    }
    """
    
    # Handle CORS preflight request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 204
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400

        filename = data.get('filename')
        project_id = data.get('Project_Id') 

        if not filename or project_id is None:
            return jsonify({"error": "Missing 'filename' or 'Project_Id' in JSON body"}), 400
        
        prediction = Prediction.query.filter_by(filename=filename).first()
        
        if not prediction:
            return jsonify({"error": f"Prediction not found for filename: {filename}"}), 404
        
        prediction.project_id = project_id
        db.session.commit()
        
        current_app.logger.info(f"Updated Project_Id for '{filename}' to {project_id}")

        return jsonify({
            "message": "Project ID updated successfully",
            "prediction": {
                "id": prediction.id,
                "filename": prediction.filename,
                "project_id": prediction.project_id
            }
        }), 200

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Error updating project ID: {str(e)}\n{traceback.format_exc()}")
        return jsonify({"error": str(e), "type": type(e).__name__}), 500
    
@cell_site_bp.route('/site-noml/<int:project_id>', methods=['GET'])
def get_site_noml_by_project(project_id):
    """
    Fetch all SiteNoMl entries for a given project_id.

    Example:
        GET /cell-site/site-noml/12345
    """
    try:
        # Query all rows with the given project_id
        from models import SiteNoMl

        sites = SiteNoMl.query.filter_by(project_id=project_id).all()

        if not sites:
            return jsonify({
                'message': f'No SiteNoMl data found for project_id {project_id}',
                'count': 0,
                'data': []
            }), 404

        # Convert SQLAlchemy objects to JSON-serializable dicts
        site_data = [{
            'id': s.id,
            'project_id': s.project_id,
            'network': s.network,
            'earfcn_or_narfcn': s.earfcn_or_narfcn,
            'site_key_inferred': s.site_key_inferred,
            'pci_or_psi': s.pci_or_psi,
            'samples': s.samples,
            'lat_pred': s.lat_pred,
            'lon_pred': s.lon_pred,
            'azimuth_deg_5': s.azimuth_deg_5,
            'azimuth_deg_5_soft': s.azimuth_deg_5_soft,
            'azimuth_deg_label_soft': s.azimuth_deg_label_soft,
            'azimuth_adjustment_deg': s.azimuth_adjustment_deg,
            'template_spacing_deg': s.template_spacing_deg,
            'beamwidth_deg_est': s.beamwidth_deg_est,
            'median_sample_distance_m': s.median_sample_distance_m,
            'cell_id_representative': s.cell_id_representative,
            'sector_count': s.sector_count,
            'azimuth_reliability': s.azimuth_reliability,
            'spacing_used': s.spacing_used
        } for s in sites]

        return jsonify({
            'project_id': project_id,
            'count': len(site_data),
            'data': site_data
        }), 200

    except Exception as e:
        current_app.logger.error(
            f"Error fetching SiteNoMl data: {str(e)}\n{traceback.format_exc()}"
        )
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500
