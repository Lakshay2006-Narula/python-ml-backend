# In tools/cell_site/routes.py

from flask import Blueprint, request, jsonify, send_file, current_app
import os
import traceback
import json
import re
import io
import csv

# Correct import order
from extensions import db
from models import Prediction, SiteNoMl

# Then service import
from .services import CellSiteService

from werkzeug.datastructures import FileStorage


# ============================================================
# 🔵 SETUP BLUEPRINT
# ============================================================
cell_site_bp = Blueprint('cell_site', __name__)


# ============================================================
# 🔵 GLOBAL CORS HANDLER (Fixes OPTIONS preflight for all routes)
# ============================================================
@cell_site_bp.before_request
def handle_preflight_for_cell_site():
    if request.method == "OPTIONS":
        response = jsonify({"status": "ok"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        return response, 204


# ============================================================
# 🔵 HEALTH CHECK
# ============================================================
@cell_site_bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'tool': 'Cell Site Locator',
        'version': '1.0.0'
    })


# ============================================================
# 🔵 UPLOAD FILE (PROCESS LOGS)
# ============================================================
@cell_site_bp.route('/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == "OPTIONS":
        return "", 204

    try:
        service = CellSiteService()

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not service.allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

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

        # Extract project_id
        project_id = request.form.get('project_id', type=int)

        # Process file
        result = service.process_file(file, params, project_id)
        if not result:
            return jsonify({'error': 'Processing failed'}), 500

        # Log result
        if result.get('output_dir') and result.get('results'):
            results_dict = result['results']
            output_filename = list(results_dict.values())[0] if results_dict else 'no_file_generated'

            new_prediction = Prediction(
                output_dir=result['output_dir'],
                filename=output_filename,
                method=params['method'],
                min_samples=params['min_samples'],
                project_id=project_id
            )
            db.session.add(new_prediction)
            db.session.commit()

        return jsonify(result), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


# ============================================================
# 🔵 SAFE HELPERS
# ============================================================
def safe_int(v):
    try: return int(float(v))
    except: return None

def safe_float(v):
    try: return float(v)
    except: return None

def extract_mci(cell_info):
    if not cell_info: return None
    m = re.search(r'mCi=([0-9*]+)', cell_info)
    return m.group(1) if m else None


# ============================================================
# 🔵 PROCESS SESSION (Convert DB logs → CSV → Run ML/NoML)
# ============================================================
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
            log.m_alpha_long,
            log.network,
            safe_int(log.earfcn),
            safe_int(log.pci),
            safe_float(log.rsrp),
            safe_float(log.rsrq),
            safe_float(log.sinr),
            log.band,
            extract_mci(log.primary_cell_info_1),
            log.ta
        ])

    csv_buffer.seek(0)
    csv_file = FileStorage(
        stream=csv_buffer,
        filename=f"project_{project_id}.csv",
        content_type="text/csv"
    )

    service = CellSiteService()

    params = {
        'method': 'noml',
        'min_samples': 30,
        'bin_size': 5,
        'soft_spacing': False,
        'use_ta': False,
        'make_map': True
    }

    result = service.process_file(csv_file, params, project_id)

    # Log
    if result and result.get('output_dir') and result.get('results'):
        output_filename = list(result['results'].values())[0]
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


# ============================================================
# 🔵 VERIFY PROJECT (FIXES YOUR FRONTEND ERROR)
# ============================================================
@cell_site_bp.route('/verify-project/<int:project_id>', methods=['GET', 'OPTIONS'])
def verify_project(project_id):
    if request.method == "OPTIONS":
        return "", 204

    try:
        result = db.session.execute(
            db.text("SELECT id FROM tbl_project WHERE id = :pid"),
            {"pid": project_id}
        ).fetchone()

        if not result:
            return jsonify({
                "Status": 0,
                "Exists": False,
                "Message": f"Project {project_id} not found"
            }), 404

        return jsonify({
            "Status": 1,
            "Exists": True,
            "Message": "Project exists",
            "project_id": project_id
        }), 200

    except Exception as e:
        return jsonify({"Status": 0, "Message": str(e)}), 500


# ============================================================
# 🔵 DOWNLOAD FILE
# ============================================================
@cell_site_bp.route('/download/<output_dir>/<filename>', methods=['GET'])
def download_file(output_dir, filename):
    try:
        safe_dir = os.path.basename(output_dir)
        safe_file = os.path.basename(filename)

        file_path = os.path.join(
            current_app.config['OUTPUT_FOLDER'], safe_dir, safe_file
        )

        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        return send_file(file_path, as_attachment=True, download_name=safe_file)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================
# 🔵 LIST OUTPUT FILES
# ============================================================
@cell_site_bp.route('/outputs/<output_dir>', methods=['GET'])
def list_outputs(output_dir):
    safe_dir = os.path.basename(output_dir)
    dir_path = os.path.join(current_app.config['OUTPUT_FOLDER'], safe_dir)

    if not os.path.isdir(dir_path):
        return jsonify({"error": "Directory not found"}), 404

    files = os.listdir(dir_path)
    return jsonify({"files": files, "count": len(files)}), 200


# ============================================================
# 🔵 UPDATE PROJECT ID FOR A PREDICTION
# ============================================================
@cell_site_bp.route('/update-project-id', methods=['POST', 'OPTIONS'])
def update_project_id():
    if request.method == "OPTIONS":
        return "", 204

    data = request.get_json()
    filename = data.get('filename')
    project_id = data.get('Project_Id')

    if not filename or project_id is None:
        return jsonify({"error": "Missing data"}), 400

    prediction = Prediction.query.filter_by(filename=filename).first()
    if not prediction:
        return jsonify({"error": "Prediction not found"}), 404

    prediction.project_id = project_id
    db.session.commit()

    return jsonify({"message": "Updated successfully"}), 200


# ============================================================
# 🔵 GET SITE-NOML BY PROJECT
# ============================================================
@cell_site_bp.route('/site-noml/<int:project_id>', methods=['GET'])
def get_site_noml_by_project(project_id):
    try:
        sites = SiteNoMl.query.filter_by(project_id=project_id).all()

        if not sites:
            return jsonify({'message': 'No data', 'count': 0, 'data': []}), 404

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
            "project_id": project_id,
            "count": len(site_data),
            "data": site_data
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
