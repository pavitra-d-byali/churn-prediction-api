import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessing import prepare_training_data, create_sample_data

def train_churn_model():
    """Train customer churn prediction model"""
    print("Preparing training data...")
    X_train, X_test, y_train, y_test, scaler, encoders = prepare_training_data()
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model and preprocessors
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': X_train.columns.tolist(),
        'accuracy': accuracy
    }
    
    with open('models/churn_model.pkl', 'wb') as f:
        pickle.dump(model_artifacts, f)
    
    print("Model saved to models/churn_model.pkl")
    
    # Save sample data for reference
    os.makedirs('data', exist_ok=True)
    sample_df = create_sample_data()
    sample_df.to_csv('data/sample_data.csv', index=False)
    print("Sample data saved to data/sample_data.csv")
    
    return model_artifacts

if __name__ == "__main__":
    train_churn_model()