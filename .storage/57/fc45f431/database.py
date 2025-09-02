import sqlite3
import json
from datetime import datetime
import os

class PredictionDatabase:
    """Simple SQLite database for storing predictions and requests"""
    
    def __init__(self, db_path='predictions.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                input_data TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                probability REAL NOT NULL,
                model_version TEXT DEFAULT 'v1.0',
                response_time_ms REAL,
                endpoint TEXT
            )
        ''')
        
        # Create requests table for API monitoring
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                endpoint TEXT NOT NULL,
                method TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                response_time_ms REAL NOT NULL,
                user_agent TEXT,
                ip_address TEXT
            )
        ''')
        
        # Create model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                model_version TEXT NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                training_samples INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_prediction(self, input_data, prediction, probability, response_time_ms=None, endpoint='/predict'):
        """Store a prediction in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (input_data, prediction, probability, response_time_ms, endpoint)
            VALUES (?, ?, ?, ?, ?)
        ''', (json.dumps(input_data), prediction, probability, response_time_ms, endpoint))
        
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        
        return prediction_id
    
    def store_api_request(self, endpoint, method, status_code, response_time_ms, user_agent=None, ip_address=None):
        """Store API request information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO api_requests (endpoint, method, status_code, response_time_ms, user_agent, ip_address)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (endpoint, method, status_code, response_time_ms, user_agent, ip_address))
        
        conn.commit()
        conn.close()
    
    def store_model_performance(self, model_version, accuracy, precision_score=None, recall_score=None, f1_score=None, training_samples=None):
        """Store model performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_performance (model_version, accuracy, precision_score, recall_score, f1_score, training_samples)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (model_version, accuracy, precision_score, recall_score, f1_score, training_samples))
        
        conn.commit()
        conn.close()
    
    def get_prediction_stats(self, days=7):
        """Get prediction statistics for the last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_predictions,
                AVG(probability) as avg_probability,
                SUM(CASE WHEN prediction = 1 THEN 1 ELSE 0 END) as churn_predictions,
                AVG(response_time_ms) as avg_response_time
            FROM predictions 
            WHERE timestamp >= datetime('now', '-{} days')
        '''.format(days))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'total_predictions': result[0],
                'avg_probability': round(result[1] or 0, 3),
                'churn_predictions': result[2],
                'churn_rate': round((result[2] / max(result[0], 1)) * 100, 2),
                'avg_response_time_ms': round(result[3] or 0, 2)
            }
        return {}
    
    def get_api_stats(self, days=7):
        """Get API usage statistics for the last N days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_requests,
                AVG(response_time_ms) as avg_response_time,
                SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_count
            FROM api_requests 
            WHERE timestamp >= datetime('now', '-{} days')
        '''.format(days))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'total_requests': result[0],
                'avg_response_time_ms': round(result[1] or 0, 2),
                'error_count': result[2],
                'error_rate': round((result[2] / max(result[0], 1)) * 100, 2)
            }
        return {}
    
    def get_recent_predictions(self, limit=10):
        """Get recent predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT timestamp, input_data, prediction, probability, endpoint
            FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'timestamp': row[0],
                'input_data': json.loads(row[1]),
                'prediction': row[2],
                'probability': row[3],
                'endpoint': row[4]
            }
            for row in results
        ]

# Global database instance
db = PredictionDatabase()