# In tools/cell_site/routes.py

from flask import Blueprint, request, jsonify, send_file, current_app
from werkzeug.utils import secure_filename
import os
import time
import traceback
from .services import CellSiteService

# === ADD THESE IMPORTS ===
from extensions import db
from models import Prediction  # Import your new model
# === END OF ADDITIONS ===


cell_site_bp = Blueprint('cell_site', __name__)
service = CellSiteService()


@cell_site_bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'tool': 'Cell Site Locator',
        'version': '1.0.0',
        'endpoints': ['/upload', '/download/<output_dir>/<filename>']
    })


@cell_site_bp.route('/upload', methods=['POST'])
def upload_file():
    """Upload and process cell site data"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not service.allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'allowed': ['csv', 'xlsx', 'xls']
            }), 400

        # Extract parameters
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

        current_app.logger.info(f"Processing file: {file.filename} with method: {params['method']}")

        # Process file
        result = service.process_file(file, params)

        # === SAVE TO 'predictions' LOG TABLE (FIXED) ===
        # We check for 'results' (a dict) instead of 'files'
        if result and result.get('output_dir') and result.get('results'):

            results_dict = result.get('results') # e.g., {'no_ta': 'file1.csv', 'soft': 'file2.csv'}

            if results_dict and isinstance(results_dict, dict) and results_dict.values():
                # Get the first filename from the results
                output_filename = list(results_dict.values())[0]
            else:
                output_filename = 'no_file_generated'

            # Create a new Prediction object
            new_prediction = Prediction(
                output_dir=result['output_dir'],
                filename=output_filename,
                method=params['method'],
                min_samples=params['min_samples']
            )

            db.session.add(new_prediction)
            db.session.commit()

            current_app.logger.info(f"Saved prediction {new_prediction.id} to log database.")
        # === END OF LOG CODE ===

        return jsonify(result), 200

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"Upload error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e), 'type': type(e).__name__}), 500

@cell_site_bp.route('/download/<output_dir>/<filename>', methods=['GET'])
def download_file(output_dir, filename):
    """Download generated files"""
    try:
        # ... (rest of your route is unchanged) ...
        
        # Local file
        file_path = os.path.join(
            current_app.config['OUTPUT_FOLDER'],
            output_dir,
            filename
        )
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        current_app.logger.error(f"Download error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@cell_site_bp.route('/outputs/<output_dir>', methods=['GET'])
def list_outputs(output_dir):
    """List all files in an output directory"""
    try:
        # ... (rest of your route is unchanged) ...
        
        dir_path = os.path.join(current_app.config['OUTPUT_FOLDER'], output_dir)
        if not os.path.exists(dir_path):
            return jsonify({'error': 'Directory not found'}), 404
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        
        return jsonify({'files': files, 'count': len(files)}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500