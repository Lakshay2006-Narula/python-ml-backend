import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', os.urandom(24).hex())
    DEBUG = False
    TESTING = False
    
    # Server (Render provides PORT)
    PORT = int(os.getenv('PORT', 10000))
    
    # Paths - Platform-specific
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Use temp dir on production (Render), local folders on dev
    if os.getenv('RENDER'):
        # Production (Render)
        UPLOAD_FOLDER = '/tmp/uploads'
        OUTPUT_FOLDER = '/tmp/outputs'
    else:
        # Development (Windows/Mac/Linux)
        UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
        OUTPUT_FOLDER = os.path.join(BASE_DIR, 'outputs')
    
    # File Upload
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))
    ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls', 'geojson', 'json'}
    
    # CORS
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    
    # External Storage
    USE_S3 = os.getenv('USE_S3', 'false').lower() == 'true'
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
    S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
    S3_REGION = os.getenv('S3_REGION', 'us-east-1')
    
    CLOUDINARY_URL = os.getenv('CLOUDINARY_URL')
    
    # OSM Settings
    OSM_TIMEOUT = int(os.getenv('OSM_TIMEOUT', 180))
    OSM_USE_CACHE = os.getenv('OSM_USE_CACHE', 'true').lower() == 'true'
    
    # Redis
    REDIS_URL = os.getenv('REDIS_URL')

    # 1. The URL string - now points to 'predicted_db'
    SQLALCHEMY_DATABASE_URI = os.getenv("DATABASE_URL")

    # 2. This disables an unneeded feature
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # 3. This passes the SSL certificate
    SQLALCHEMY_ENGINE_OPTIONS = {
        'connect_args': {
            'ssl': {
                'ca': os.path.join(BASE_DIR, 'ca.pem')
            }
        }
    }
    # === END OF DATABASE CONFIGURATION ===
    
    # Tool configs
    CELL_SITE_MIN_SAMPLES = int(os.getenv('CELL_SITE_MIN_SAMPLES', 30))
    CELL_SITE_BIN_SIZE = int(os.getenv('CELL_SITE_BIN_SIZE', 5))
    
    @staticmethod
    def init_app():
        """Initialize application directories"""
        for folder in [Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER]:
            os.makedirs(folder, exist_ok=True)
            print(f"✅ Created directory: {folder}")

class DevelopmentConfig(Config):
    DEBUG = True
    # Already using local folders from parent class

class ProductionConfig(Config):
    DEBUG = False

class RenderConfig(ProductionConfig):
    """Render-specific configuration"""
    pass

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'render': RenderConfig,
    'default': DevelopmentConfig  # Changed to Development for local
}