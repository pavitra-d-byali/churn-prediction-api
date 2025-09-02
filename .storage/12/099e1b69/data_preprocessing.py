import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def create_sample_data():
    """Create sample customer churn data for training"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(40, 15, n_samples).astype(int),
        'tenure': np.random.exponential(2, n_samples),
        'monthly_charges': np.random.normal(65, 20, n_samples),
        'total_charges': np.random.normal(2000, 1000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
        'online_security': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'tech_support': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
    }
    
    df = pd.DataFrame(data)
    
    # Create churn target based on some logic
    churn_prob = (
        0.1 + 
        0.3 * (df['contract_type'] == 'Month-to-month') +
        0.2 * (df['monthly_charges'] > 80) +
        0.15 * (df['tenure'] < 1) +
        0.1 * (df['online_security'] == 'No') +
        0.1 * (df['tech_support'] == 'No')
    )
    
    df['churn'] = np.random.binomial(1, churn_prob)
    
    return df

def preprocess_features(df, scaler=None, encoders=None, fit_transform=True):
    """Preprocess features for model training/prediction"""
    df_processed = df.copy()
    
    # Remove customer_id if present
    if 'customer_id' in df_processed.columns:
        df_processed = df_processed.drop('customer_id', axis=1)
    
    # Separate numerical and categorical features
    numerical_features = ['age', 'tenure', 'monthly_charges', 'total_charges']
    categorical_features = ['contract_type', 'payment_method', 'internet_service', 'online_security', 'tech_support']
    
    if fit_transform:
        # Initialize scalers and encoders
        scaler = StandardScaler()
        encoders = {}
        
        # Scale numerical features
        df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
        
        # Encode categorical features
        for feature in categorical_features:
            encoder = LabelEncoder()
            df_processed[feature] = encoder.fit_transform(df_processed[feature])
            encoders[feature] = encoder
            
        return df_processed, scaler, encoders
    else:
        # Use existing scalers and encoders
        df_processed[numerical_features] = scaler.transform(df_processed[numerical_features])
        
        for feature in categorical_features:
            df_processed[feature] = encoders[feature].transform(df_processed[feature])
            
        return df_processed

def prepare_training_data():
    """Prepare data for model training"""
    df = create_sample_data()
    
    # Separate features and target
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Preprocess features
    X_processed, scaler, encoders = preprocess_features(X, fit_transform=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler, encoders