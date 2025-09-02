# 🤖 Customer Churn Prediction MLOps API

A production-ready machine learning API for predicting customer churn using Flask, scikit-learn, and comprehensive monitoring.

## 🚀 Features

- **Machine Learning Model**: RandomForest classifier with 61% accuracy
- **RESTful API**: Multiple endpoints for predictions and monitoring
- **Real-time Monitoring**: API usage metrics, request logging, and error tracking
- **Database Integration**: SQLite for storing predictions and metrics
- **Production Ready**: CI/CD pipeline, testing, and Docker support
- **Interactive Web Interface**: Built-in form for testing predictions

## 📊 API Endpoints

- `GET /` - Interactive web interface with API documentation
- `GET /health` - Health check endpoint
- `GET /metrics` - Real-time API usage metrics
- `GET /model/info` - Model information and accuracy
- `POST /predict` - Single customer churn prediction
- `POST /batch_predict` - Batch predictions for multiple customers
- `POST /retrain` - Retrain the model (development feature)

## 🛠️ Local Development

### Prerequisites
- Python 3.10+
- pip

### Setup
```bash
# Clone the repository
git clone <your-repo-url>
cd flask_template

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

The application will start on a random available port and display the URL.

## 🌐 Cloud Deployment

### Option 1: Deploy on Render (Recommended)

1. Push your code to GitHub
2. Go to [Render](https://render.com)
3. Create a new Web Service
4. Connect your GitHub repository
5. Use these settings:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --config gunicorn_config.py`

### Option 2: Deploy on Railway

1. Push your code to GitHub (with Dockerfile)
2. Go to [Railway](https://railway.app)
3. Create new project from GitHub repo
4. Railway will auto-detect and deploy

### Option 3: Docker Deployment

```bash
# Build the image
docker build -t churn-prediction-api .

# Run the container
docker run -p 5000:5000 churn-prediction-api
```

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=html
```

## 📈 Model Information

- **Algorithm**: RandomForest Classifier
- **Accuracy**: 61%
- **Features**: 9 customer attributes (age, tenure, charges, contract type, etc.)
- **Training Data**: Synthetic customer data with churn labels

## 🔧 Configuration

Environment variables:
- `FLASK_ENV`: Set to 'production' for production deployment
- `SECRET_KEY`: Flask secret key (auto-generated in development)
- `DATABASE_URL`: Database connection string (SQLite by default)
- `PORT`: Port number (auto-assigned by cloud platforms)

## 📊 Monitoring

The application includes comprehensive monitoring:
- Request counting and timing
- Error rate tracking
- Prediction logging
- API usage metrics
- Real-time uptime monitoring

Access metrics at `/metrics` endpoint.

## 🔄 CI/CD Pipeline

GitHub Actions workflow includes:
- Multi-version Python testing (3.8, 3.9, 3.10)
- Code quality checks with flake8
- Security scanning with Trivy
- Docker image building
- Automated testing on pull requests

## 📝 API Usage Examples

### Single Prediction
```bash
curl -X POST http://your-app-url/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "tenure": 2.5,
    "monthly_charges": 75.0,
    "total_charges": 1875.0,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "tech_support": "No"
  }'
```

### Batch Prediction
```bash
curl -X POST http://your-app-url/batch_predict \
  -H "Content-Type: application/json" \
  -d '[
    {"age": 35, "tenure": 2.5, ...},
    {"age": 42, "tenure": 1.2, ...}
  ]'
```

## 🏗️ Architecture

```
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── routes.py            # API endpoints
│   ├── monitoring.py        # Request monitoring
│   └── database.py          # Database models
├── ml_model/
│   ├── train_model.py       # Model training
│   ├── model_utils.py       # Prediction utilities
│   └── data_preprocessing.py # Data processing
├── models/
│   └── churn_model.pkl      # Trained model
├── tests/
│   ├── test_api.py          # API tests
│   └── test_model.py        # Model tests
├── .github/workflows/
│   └── ci-cd.yml           # CI/CD pipeline
├── Dockerfile              # Docker configuration
├── requirements.txt        # Python dependencies
├── gunicorn_config.py     # Production server config
└── app.py                 # Application entry point
```

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## 📞 Support

For issues and questions, please open a GitHub issue or contact the development team.