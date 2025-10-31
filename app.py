from flask import Flask, jsonify
from flask_cors import CORS
import os

# 1. Import config, blueprints, AND the db extension
from config import config
from tools.buildings.routes import buildings_bp
from tools.cell_site.routes import cell_site_bp
from extensions import db  # <-- ADD THIS IMPORT

def create_app(config_name='default'):
    """
    Flask app factory pattern.
    """
    app = Flask(__name__)
    
    # 2. Load configuration
    # Finds config_name in your config.py (e.g., 'development' from FLASK_ENV)
    env_config = config.get(config_name, config['default'])
    app.config.from_object(env_config)
    
    # Create upload/output folders if they don't exist
    # This calls the init_app() method from your config.py
    if hasattr(env_config, 'init_app'):
        env_config.init_app() 

    # 3. Initialize Extensions
    db.init_app(app)  # <-- ADD THIS LINE

    # 4. Enable CORS
    # Uses the CORS_ORIGINS list from your config
    CORS(app, resources={r"/api/*": {"origins": app.config.get('CORS_ORIGINS', '*')}})

    # 5. Register Blueprints with a URL prefix
    # This makes your /api/buildings routes active
    app.register_blueprint(buildings_bp, url_prefix='/api/buildings')
    # This makes your /api/cell-site routes active
    app.register_blueprint(cell_site_bp, url_prefix='/api/cell-site')

    # 6. Define Root and Health Check Endpoints
    
    # Root endpoint
    @app.route('/', methods=['GET'])
    def root():
        return jsonify({
            "message": "Cell Site Locator API is running!"
        })

    # Main health check for Render/monitoring
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint for monitoring"""
        return jsonify({
            'status': 'healthy',
            'service': 'Python ML Backend',
            'message': 'Cell Site Locator API is running!'
        }), 200
    
    return app

# --- Application Entry Point ---

# Get the environment (e.g., 'development') from the .env file
app_env = os.getenv('FLASK_ENV', 'default')

# Create the app instance
app = create_app(app_env)

if __name__ == '__main__':
    # This block runs when you execute `python app.py`
    
    # Get the port from the .env file, with 8080 as a fallback
    port = int(os.getenv('PORT', 8080))
    
    app.run(
        host=os.getenv('HOST', '0.0.0.0'), 
        port=port, 
        debug=app.config.get('DEBUG', False)
    )