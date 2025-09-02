import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///churn_predictions.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Production settings
    if os.environ.get('FLASK_ENV') == 'production':
        DEBUG = False
        TESTING = False
    else:
        DEBUG = True
        TESTING = False

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    
class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'