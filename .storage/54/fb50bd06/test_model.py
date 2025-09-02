import pytest
import sys
import os
import numpy as np

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ml_model.model_utils import ChurnPredictor, validate_customer_data
from ml_model.data_preprocessing import create_sample_data, preprocess_features

class TestChurnPredictor:
    """Test ChurnPredictor class"""
    
    @pytest.fixture
    def predictor(self):
        """Create a ChurnPredictor instance"""
        return ChurnPredictor()
    
    @pytest.fixture
    def sample_customer(self):
        """Sample customer data"""
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
    
    def test_predictor_initialization(self, predictor):
        """Test that predictor initializes correctly"""
        assert predictor is not None
        assert predictor.model_artifacts is not None
        assert 'model' in predictor.model_artifacts
        assert 'scaler' in predictor.model_artifacts
        assert 'encoder' in predictor.model_artifacts
    
    def test_get_model_info(self, predictor):
        """Test get_model_info method"""
        info = predictor.get_model_info()
        
        assert 'model_type' in info
        assert 'features' in info
        assert 'accuracy' in info
        assert isinstance(info['features'], list)
        assert len(info['features']) > 0
    
    def test_predict_single(self, predictor, sample_customer):
        """Test single prediction"""
        result = predictor.predict_single(sample_customer)
        
        assert 'churn_prediction' in result
        assert 'churn_probability' in result
        assert result['churn_prediction'] in [0, 1]
        assert 0 <= result['churn_probability'] <= 1
        assert isinstance(result['churn_prediction'], (int, np.integer))
        assert isinstance(result['churn_probability'], (float, np.floating))
    
    def test_predict_batch(self, predictor, sample_customer):
        """Test batch prediction"""
        batch_data = [sample_customer, sample_customer.copy()]
        results = predictor.predict_batch(batch_data)
        
        assert len(results) == 2
        for result in results:
            assert 'churn_prediction' in result
            assert 'churn_probability' in result
            assert result['churn_prediction'] in [0, 1]
            assert 0 <= result['churn_probability'] <= 1

class TestDataValidation:
    """Test data validation functions"""
    
    def test_validate_customer_data_valid(self):
        """Test validation with valid customer data"""
        valid_data = {
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
        
        is_valid, message = validate_customer_data(valid_data)
        assert is_valid is True
        assert message == "Valid"
    
    def test_validate_customer_data_missing_fields(self):
        """Test validation with missing required fields"""
        invalid_data = {
            "age": 35,
            "tenure": 2.5
            # Missing other required fields
        }
        
        is_valid, message = validate_customer_data(invalid_data)
        assert is_valid is False
        assert "Missing required field" in message
    
    def test_validate_customer_data_invalid_types(self):
        """Test validation with invalid data types"""
        invalid_data = {
            "age": "thirty-five",  # Should be numeric
            "tenure": 2.5,
            "monthly_charges": 75.0,
            "total_charges": 1875.0,
            "contract_type": "Month-to-month",
            "payment_method": "Electronic check",
            "internet_service": "Fiber optic",
            "online_security": "No",
            "tech_support": "No"
        }
        
        is_valid, message = validate_customer_data(invalid_data)
        assert is_valid is False
        assert "must be numeric" in message
    
    def test_validate_customer_data_invalid_values(self):
        """Test validation with invalid categorical values"""
        invalid_data = {
            "age": 35,
            "tenure": 2.5,
            "monthly_charges": 75.0,
            "total_charges": 1875.0,
            "contract_type": "Invalid Contract",  # Invalid value
            "payment_method": "Electronic check",
            "internet_service": "Fiber optic",
            "online_security": "No",
            "tech_support": "No"
        }
        
        is_valid, message = validate_customer_data(invalid_data)
        assert is_valid is False
        assert "Invalid value" in message

class TestDataPreprocessing:
    """Test data preprocessing functions"""
    
    def test_create_sample_data(self):
        """Test sample data creation"""
        df = create_sample_data(n_samples=100)
        
        assert len(df) == 100
        assert 'age' in df.columns
        assert 'tenure' in df.columns
        assert 'churn' in df.columns
        
        # Check data types and ranges
        assert df['age'].dtype in ['int64', 'float64']
        assert df['tenure'].dtype in ['int64', 'float64']
        assert df['churn'].dtype in ['int64', 'bool']
        
        # Check value ranges
        assert df['age'].min() >= 18
        assert df['age'].max() <= 80
        assert df['tenure'].min() >= 0
        assert df['churn'].isin([0, 1]).all()
    
    def test_preprocess_features(self):
        """Test feature preprocessing"""
        df = create_sample_data(n_samples=50)
        
        # Test preprocessing
        X_processed, scaler, encoder = preprocess_features(df, fit_transformers=True)
        
        assert X_processed is not None
        assert scaler is not None
        assert encoder is not None
        assert len(X_processed) == 50
        
        # Test that processed features are numeric
        assert X_processed.dtype in ['float64', 'float32']
        
        # Test preprocessing with existing transformers
        df_new = create_sample_data(n_samples=10)
        X_new = preprocess_features(df_new, scaler=scaler, encoder=encoder, fit_transformers=False)
        
        assert X_new is not None
        assert len(X_new) == 10
        assert X_new.shape[1] == X_processed.shape[1]  # Same number of features

if __name__ == '__main__':
    pytest.main([__file__])