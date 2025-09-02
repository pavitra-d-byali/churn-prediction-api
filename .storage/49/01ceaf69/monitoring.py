import logging
import time
from functools import wraps
from flask import request, jsonify
import json
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class APIMonitor:
    def __init__(self):
        self.request_count = 0
        self.prediction_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
    def log_request(self, endpoint, method, status_code, response_time):
        """Log API request details"""
        self.request_count += 1
        
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'response_time_ms': round(response_time * 1000, 2),
            'request_count': self.request_count
        }
        
        logger.info(f"API_REQUEST: {json.dumps(log_data)}")
        
        if status_code >= 400:
            self.error_count += 1
            
    def log_prediction(self, input_data, prediction, confidence=None):
        """Log prediction details"""
        self.prediction_count += 1
        
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'prediction': prediction,
            'confidence': confidence,
            'input_features': input_data,
            'prediction_count': self.prediction_count
        }
        
        logger.info(f"PREDICTION: {json.dumps(log_data)}")
        
    def get_metrics(self):
        """Get current metrics"""
        uptime = time.time() - self.start_time
        
        return {
            'uptime_seconds': round(uptime, 2),
            'total_requests': self.request_count,
            'total_predictions': self.prediction_count,
            'error_count': self.error_count,
            'error_rate': round(self.error_count / max(self.request_count, 1) * 100, 2),
            'requests_per_minute': round(self.request_count / (uptime / 60), 2) if uptime > 0 else 0
        }

# Global monitor instance
monitor = APIMonitor()

def monitor_requests(f):
    """Decorator to monitor API requests"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        start_time = time.time()
        
        try:
            response = f(*args, **kwargs)
            status_code = getattr(response, 'status_code', 200)
            
            # Log successful request
            monitor.log_request(
                endpoint=request.endpoint,
                method=request.method,
                status_code=status_code,
                response_time=time.time() - start_time
            )
            
            return response
            
        except Exception as e:
            # Log error
            monitor.log_request(
                endpoint=request.endpoint,
                method=request.method,
                status_code=500,
                response_time=time.time() - start_time
            )
            
            logger.error(f"ERROR in {request.endpoint}: {str(e)}")
            raise
            
    return decorated_function