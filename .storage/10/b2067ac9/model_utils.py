import pickle
import pandas as pd
import numpy as np
from ml_model.data_preprocessing import preprocess_features

class ChurnPredictor:
    def __init__(self, model_path='models/churn_model.pkl'):
        """Initialize the churn predictor with trained model"""
        self.model_path = model_path
        self.model_artifacts = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and preprocessors"""
        try:
            with open(self.model_path, 'rb') as f:
                self.model_artifacts = pickle.load(f)
            print(f"Model loaded successfully. Accuracy: {self.model_artifacts['accuracy']:.4f}")
        except FileNotFoundError:
            print(f"Model file not found at {self.model_path}. Please train the model first.")
            self.model_artifacts = None
    
    def predict_single(self, customer_data):
        """Predict churn for a single customer"""
        if self.model_artifacts is None:
            raise ValueError("Model not loaded. Please train the model first.")
        
        # Convert to DataFrame if it's a dict
        if isinstance(customer_data, dict):
            df = pd.DataFrame([customer_data])
        else:
            df = customer_data.copy()
        
        # Preprocess the data
        df_processed = preprocess_features(
            df, 
            scaler=self.model_artifacts['scaler'],
            encoders=self.model_artifacts['encoders'],
            fit_transform=False
        )
        
        # Make prediction
        prediction = self.model_artifacts['model'].predict(df_processed)[0]
        probability = self.model_artifacts['model'].predict_proba(df_processed)[0]
        
        return {
            'churn_prediction': int(prediction),
            'churn_probability': float(probability[1]),
            'no_churn_probability': float(probability[0])
        }
    
    def predict_batch(self, customers_data):
        """Predict churn for multiple customers"""
        if self.model_artifacts is None:
            raise ValueError("Model not loaded. Please train the model first.")
        
        # Convert to DataFrame if it's a list of dicts
        if isinstance(customers_data, list):
            df = pd.DataFrame(customers_data)
        else:
            df = customers_data.copy()
        
        # Preprocess the data
        df_processed = preprocess_features(
            df,
            scaler=self.model_artifacts['scaler'],
            encoders=self.model_artifacts['encoders'],
            fit_transform=False
        )
        
        # Make predictions
        predictions = self.model_artifacts['model'].predict(df_processed)
        probabilities = self.model_artifacts['model'].predict_proba(df_processed)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            results.append({
                'customer_index': i,
                'churn_prediction': int(pred),
                'churn_probability': float(prob[1]),
                'no_churn_probability': float(prob[0])
            })
        
        return results
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model_artifacts is None:
            return {"error": "Model not loaded"}
        
        return {
            "model_type": "RandomForestClassifier",
            "accuracy": self.model_artifacts['accuracy'],
            "feature_names": self.model_artifacts['feature_names'],
            "model_loaded": True
        }

def validate_customer_data(data):
    """Validate customer data format"""
    required_fields = [
        'age', 'tenure', 'monthly_charges', 'total_charges',
        'contract_type', 'payment_method', 'internet_service',
        'online_security', 'tech_support'
    ]
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {missing_fields}"
    
    # Validate data types and ranges
    try:
        age = int(data['age'])
        if age < 18 or age > 100:
            return False, "Age must be between 18 and 100"
        
        tenure = float(data['tenure'])
        if tenure < 0:
            return False, "Tenure must be non-negative"
        
        monthly_charges = float(data['monthly_charges'])
        if monthly_charges <= 0:
            return False, "Monthly charges must be positive"
        
        total_charges = float(data['total_charges'])
        if total_charges < 0:
            return False, "Total charges must be non-negative"
        
        # Validate categorical fields
        valid_contract_types = ['Month-to-month', 'One year', 'Two year']
        if data['contract_type'] not in valid_contract_types:
            return False, f"Contract type must be one of: {valid_contract_types}"
        
        valid_payment_methods = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
        if data['payment_method'] not in valid_payment_methods:
            return False, f"Payment method must be one of: {valid_payment_methods}"
        
        valid_internet_services = ['DSL', 'Fiber optic', 'No']
        if data['internet_service'] not in valid_internet_services:
            return False, f"Internet service must be one of: {valid_internet_services}"
        
        valid_yes_no = ['Yes', 'No']
        if data['online_security'] not in valid_yes_no:
            return False, "Online security must be 'Yes' or 'No'"
        
        if data['tech_support'] not in valid_yes_no:
            return False, "Tech support must be 'Yes' or 'No'"
        
    except (ValueError, TypeError) as e:
        return False, f"Invalid data type: {str(e)}"
    
    return True, "Valid"