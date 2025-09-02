import os
import socket
from app import create_app
from config import Config, ProductionConfig, DevelopmentConfig

# Determine configuration based on environment
if os.environ.get('FLASK_ENV') == 'production':
    config_class = ProductionConfig
else:
    config_class = DevelopmentConfig

app = create_app(config_class)

def find_free_port():
    """Find a free port for development"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

if __name__ == '__main__':
    # For development only
    if not os.environ.get('FLASK_ENV') == 'production':
        port = int(os.environ.get('PORT', find_free_port()))
        print(f"Starting Flask app on port {port}")
        app.run(host='0.0.0.0', port=port, debug=True)
    else:
        # In production, use gunicorn
        print("Production mode - use gunicorn to run the app")