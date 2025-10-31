from flask import Blueprint, request, jsonify, current_app
import traceback
from .services import BuildingService

buildings_bp = Blueprint('buildings', __name__)
service = BuildingService()

@buildings_bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'tool': 'Building Extraction',
        'version': '1.0.0',
        'endpoints': ['/generate', '/test']
    })

@buildings_bp.route('/generate', methods=['POST'])
def generate_buildings():
    """Generate buildings from OpenStreetMap"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'Status': 0,
                'Message': 'No data provided'
            }), 400
        
        current_app.logger.info(f"Building extraction request: {data.keys()}")
        
        result = service.extract_buildings(data)
        
        return jsonify(result), 200
        
    except Exception as e:
        current_app.logger.error(f"Building extraction error: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            'Status': 0,
            'Message': f'Error: {str(e)}'
        }), 500

@buildings_bp.route('/test', methods=['GET'])
def test():
    """Test endpoint with sample data"""
    sample = {
        "WKT": "POLYGON((77.2090 28.6139, 77.2100 28.6139, 77.2100 28.6149, 77.2090 28.6149, 77.2090 28.6139))"
    }
    
    try:
        result = service.extract_buildings(sample)
        return jsonify(result), 200
    except Exception as e:
        current_app.logger.error(f"Test error: {str(e)}")
        return jsonify({'Status': 0, 'Message': str(e)}), 500