from flask import Blueprint, jsonify, request, render_template_string
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ml_model.model_utils import ChurnPredictor, validate_customer_data

main_bp = Blueprint('main', __name__)

# Initialize the predictor
predictor = ChurnPredictor()

@main_bp.route('/')
def index():
    """Home page with API documentation"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Customer Churn Prediction API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            .endpoint { background: #f8f9fa; padding: 15px; margin: 15px 0; border-radius: 5px; border-left: 4px solid #007bff; }
            .method { background: #007bff; color: white; padding: 4px 8px; border-radius: 3px; font-size: 12px; }
            pre { background: #f1f1f1; padding: 10px; border-radius: 3px; overflow-x: auto; }
            .test-form { background: #e9ecef; padding: 20px; border-radius: 5px; margin: 20px 0; }
            input, select { margin: 5px; padding: 8px; border: 1px solid #ddd; border-radius: 3px; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 3px; cursor: pointer; }
            button:hover { background: #0056b3; }
            #result { margin-top: 15px; padding: 10px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 Customer Churn Prediction API</h1>
            <p>MLOps project demonstrating machine learning model deployment with Flask.</p>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /health</h3>
                <p>Check API health status</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">GET</span> /model/info</h3>
                <p>Get information about the loaded model</p>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /predict</h3>
                <p>Predict churn for a single customer</p>
                <pre>{
  "age": 35,
  "tenure": 2.5,
  "monthly_charges": 75.0,
  "total_charges": 1875.0,
  "contract_type": "Month-to-month",
  "payment_method": "Electronic check",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "tech_support": "No"
}</pre>
            </div>
            
            <div class="endpoint">
                <h3><span class="method">POST</span> /batch_predict</h3>
                <p>Predict churn for multiple customers (send array of customer objects)</p>
            </div>
            
            <div class="test-form">
                <h3>🧪 Test Single Prediction</h3>
                <form id="predictionForm">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                        <div>
                            <label>Age:</label><br>
                            <input type="number" id="age" value="35" min="18" max="100" required>
                        </div>
                        <div>
                            <label>Tenure (years):</label><br>
                            <input type="number" id="tenure" value="2.5" step="0.1" min="0" required>
                        </div>
                        <div>
                            <label>Monthly Charges:</label><br>
                            <input type="number" id="monthly_charges" value="75.0" step="0.01" min="0" required>
                        </div>
                        <div>
                            <label>Total Charges:</label><br>
                            <input type="number" id="total_charges" value="1875.0" step="0.01" min="0" required>
                        </div>
                        <div>
                            <label>Contract Type:</label><br>
                            <select id="contract_type" required>
                                <option value="Month-to-month">Month-to-month</option>
                                <option value="One year">One year</option>
                                <option value="Two year">Two year</option>
                            </select>
                        </div>
                        <div>
                            <label>Payment Method:</label><br>
                            <select id="payment_method" required>
                                <option value="Electronic check">Electronic check</option>
                                <option value="Mailed check">Mailed check</option>
                                <option value="Bank transfer">Bank transfer</option>
                                <option value="Credit card">Credit card</option>
                            </select>
                        </div>
                        <div>
                            <label>Internet Service:</label><br>
                            <select id="internet_service" required>
                                <option value="DSL">DSL</option>
                                <option value="Fiber optic">Fiber optic</option>
                                <option value="No">No</option>
                            </select>
                        </div>
                        <div>
                            <label>Online Security:</label><br>
                            <select id="online_security" required>
                                <option value="Yes">Yes</option>
                                <option value="No">No</option>
                            </select>
                        </div>
                        <div>
                            <label>Tech Support:</label><br>
                            <select id="tech_support" required>
                                <option value="Yes">Yes</option>
                                <option value="No">No</option>
                            </select>
                        </div>
                    </div>
                    <br>
                    <button type="submit">Predict Churn</button>
                </form>
                <div id="result"></div>
            </div>
        </div>
        
        <script>
            document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const formData = {
                    age: parseInt(document.getElementById('age').value),
                    tenure: parseFloat(document.getElementById('tenure').value),
                    monthly_charges: parseFloat(document.getElementById('monthly_charges').value),
                    total_charges: parseFloat(document.getElementById('total_charges').value),
                    contract_type: document.getElementById('contract_type').value,
                    payment_method: document.getElementById('payment_method').value,
                    internet_service: document.getElementById('internet_service').value,
                    online_security: document.getElementById('online_security').value,
                    tech_support: document.getElementById('tech_support').value
                };
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(formData)
                    });
                    
                    const result = await response.json();
                    const resultDiv = document.getElementById('result');
                    
                    if (response.ok) {
                        const churnProb = (result.churn_probability * 100).toFixed(1);
                        const prediction = result.churn_prediction === 1 ? 'WILL CHURN' : 'WILL NOT CHURN';
                        const color = result.churn_prediction === 1 ? '#dc3545' : '#28a745';
                        
                        resultDiv.innerHTML = `
                            <div style="background: ${color}; color: white; padding: 10px; border-radius: 3px;">
                                <strong>Prediction: ${prediction}</strong><br>
                                Churn Probability: ${churnProb}%
                            </div>
                        `;
                    } else {
                        resultDiv.innerHTML = `<div style="background: #dc3545; color: white; padding: 10px; border-radius: 3px;">Error: ${result.error}</div>`;
                    }
                } catch (error) {
                    document.getElementById('result').innerHTML = `<div style="background: #dc3545; color: white; padding: 10px; border-radius: 3px;">Error: ${error.message}</div>`;
                }
            });
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@main_bp.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "Customer Churn Prediction API",
        "model_loaded": predictor.model_artifacts is not None
    })

@main_bp.route('/model/info')
def model_info():
    """Get model information"""
    return jsonify(predictor.get_model_info())

@main_bp.route('/predict', methods=['POST'])
def predict_single():
    """Predict churn for a single customer"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate input data
        is_valid, message = validate_customer_data(data)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        # Make prediction
        result = predictor.predict_single(data)
        
        return jsonify({
            "success": True,
            "prediction": result,
            "input_data": data
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@main_bp.route('/batch_predict', methods=['POST'])
def predict_batch():
    """Predict churn for multiple customers"""
    try:
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({"error": "Data must be a list of customer objects"}), 400
        
        if len(data) > 100:
            return jsonify({"error": "Maximum 100 customers per batch"}), 400
        
        # Validate each customer data
        for i, customer in enumerate(data):
            is_valid, message = validate_customer_data(customer)
            if not is_valid:
                return jsonify({"error": f"Customer {i}: {message}"}), 400
        
        # Make predictions
        results = predictor.predict_batch(data)
        
        return jsonify({
            "success": True,
            "predictions": results,
            "total_customers": len(data)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@main_bp.route('/retrain', methods=['POST'])
def retrain_model():
    """Retrain the model (in production, this would be more sophisticated)"""
    try:
        from ml_model.train_model import train_churn_model
        
        # Retrain model
        model_artifacts = train_churn_model()
        
        # Reload predictor
        predictor.load_model()
        
        return jsonify({
            "success": True,
            "message": "Model retrained successfully",
            "new_accuracy": model_artifacts['accuracy']
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500