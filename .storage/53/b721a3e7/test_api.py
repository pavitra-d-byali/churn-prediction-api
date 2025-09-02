import pytest
import json
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app

@pytest.fixture
def client():
    """Create a test client for the Flask application"""
    app = create_app()
    app.config['TESTING'] = True
    
    with app.test_client() as client:
        with app.app_context():
            yield client

@pytest.fixture
def sample_customer_data():
    """Sample customer data for testing"""
    return {
        "age": 35,
        "tenure": 2.5,
        "monthly_charges": 75.0,
        "total_charges": 1875.0,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
        "online_security": "No",
        "tech_support": "No"
    }

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_endpoint(self, client):
        """Test health endpoint returns correct status"""
        response = client.get('/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['status'] == 'healthy'
        assert data['service'] == 'Customer Churn Prediction API'
        assert 'model_loaded' in data

class TestModelInfoEndpoint:
    """Test model info endpoint"""
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint returns model information"""
        response = client.get('/model/info')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'model_type' in data
        assert 'features' in data
        assert 'accuracy' in data

class TestMetricsEndpoint:
    """Test metrics endpoint"""
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint returns monitoring data"""
        response = client.get('/metrics')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'metrics' in data
        assert 'uptime_seconds' in data['metrics']
        assert 'total_requests' in data['metrics']

class TestPredictionEndpoint:
    """Test prediction endpoints"""
    
    def test_predict_single_valid_data(self, client, sample_customer_data):
        """Test single prediction with valid data"""
        response = client.post('/predict',
                             data=json.dumps(sample_customer_data),
                             content_type='application/json')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'prediction' in data
        assert 'churn_prediction' in data['prediction']
        assert 'churn_probability' in data['prediction']
        
        # Check prediction values are in valid range
        assert data['prediction']['churn_prediction'] in [0, 1]
        assert 0 <= data['prediction']['churn_probability'] <= 1
    
    def test_predict_single_missing_data(self, client):
        """Test single prediction with missing data"""
        incomplete_data = {
            "age": 35,
            "tenure": 2.5
            # Missing required fields
        }
        
        response = client.post('/predict',
                             data=json.dumps(incomplete_data),
                             content_type='application/json')
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_predict_single_no_data(self, client):
        """Test single prediction with no data"""
        response = client.post('/predict',
                             data=json.dumps({}),
                             content_type='application/json')
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_predict_batch_valid_data(self, client, sample_customer_data):
        """Test batch prediction with valid data"""
        batch_data = [sample_customer_data, sample_customer_data.copy()]
        
        response = client.post('/batch_predict',
                             data=json.dumps(batch_data),
                             content_type='application/json')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert data['success'] is True
        assert 'predictions' in data
        assert len(data['predictions']) == 2
        assert data['total_customers'] == 2
    
    def test_predict_batch_too_many_customers(self, client, sample_customer_data):
        """Test batch prediction with too many customers"""
        batch_data = [sample_customer_data] * 101  # Exceed limit of 100
        
        response = client.post('/batch_predict',
                             data=json.dumps(batch_data),
                             content_type='application/json')
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Maximum 100 customers' in data['error']
    
    def test_predict_batch_invalid_format(self, client, sample_customer_data):
        """Test batch prediction with invalid format (not a list)"""
        response = client.post('/batch_predict',
                             data=json.dumps(sample_customer_data),  # Single object instead of list
                             content_type='application/json')
        
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
        assert 'must be a list' in data['error']

class TestIndexEndpoint:
    """Test index/home endpoint"""
    
    def test_index_endpoint(self, client):
        """Test index endpoint returns HTML documentation"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Customer Churn Prediction API' in response.data
        assert b'text/html' in response.content_type.encode()

if __name__ == '__main__':
    pytest.main([__file__])