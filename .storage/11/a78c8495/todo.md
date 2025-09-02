# MLOps Project - Customer Churn Prediction MVP

## Overview
Build a customer churn prediction model and deploy it as a Flask web service with Docker containerization.

## Files to Create/Modify:

### 1. Machine Learning Components
- `ml_model/train_model.py` - Train customer churn prediction model
- `ml_model/model_utils.py` - Model loading and prediction utilities
- `ml_model/data_preprocessing.py` - Data preprocessing functions

### 2. Flask API
- `app/routes.py` - Update with ML prediction endpoints
- `app/models.py` - Update with prediction request/response models
- `app/__init__.py` - Update Flask app configuration

### 3. Data & Model Storage
- `data/sample_data.csv` - Sample customer data for training
- `models/churn_model.pkl` - Trained model file (generated)

### 4. Deployment
- `Dockerfile` - Container configuration
- `requirements.txt` - Update with ML dependencies
- `docker-compose.yml` - Local development setup

## Implementation Strategy:
1. Create simple customer churn dataset
2. Train basic RandomForest model
3. Create Flask API endpoints for predictions
4. Add Docker containerization
5. Test the complete pipeline

## Key Features:
- `/predict` endpoint for single predictions
- `/batch_predict` endpoint for multiple predictions
- Model health check endpoint
- Simple web interface for testing