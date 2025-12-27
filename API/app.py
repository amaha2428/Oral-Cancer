from flask import Flask, request, jsonify, send_from_directory, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import joblib
import pickle
import numpy as np
import pandas as pd
import logging
import time
import uuid
from datetime import datetime
from functools import wraps

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_usage.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Load trained model
try:
    model = joblib.load("random_forest_model.joblib")
    print(model)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

# Feature configuration
FEATURE_ORDER = [
    "age", "gender", "tobacco", "alcohol", "hpv", "betel_quid",
    "sun_exposure", "oral_hygiene", "diet_quality", "family_history",
    "immune_compromised", "oral_lesions", "bleeding", "swallowing",
    "mouth_patches", "region"
]

FEATURE_VALIDATION = {
    "age": {"min": 0, "max": 120, "type": "int"},
    "gender": {"min": 0, "max": 1, "type": "int"},
    "tobacco": {"min": 0, "max": 1, "type": "int"},
    "alcohol": {"min": 0, "max": 1, "type": "int"},
    "hpv": {"min": 0, "max": 1, "type": "int"},
    "betel_quid": {"min": 0, "max": 1, "type": "int"},
    "sun_exposure": {"min": 0, "max": 1, "type": "int"},
    "oral_hygiene": {"min": 0, "max": 1, "type": "int"},
    "diet_quality": {"min": 0, "max": 2, "type": "int"},
    "family_history": {"min": 0, "max": 1, "type": "int"},
    "immune_compromised": {"min": 0, "max": 1, "type": "int"},
    "oral_lesions": {"min": 0, "max": 1, "type": "int"},
    "bleeding": {"min": 0, "max": 1, "type": "int"},
    "swallowing": {"min": 0, "max": 1, "type": "int"},
    "mouth_patches": {"min": 0, "max": 1, "type": "int"},
    "region": {"min": 0, "max": 10, "type": "int"}
}

# Middleware: Request tracking
@app.before_request
def before_request():
    g.start_time = time.time()
    g.request_id = str(uuid.uuid4())
    logger.info(f"Request {g.request_id}: {request.method} {request.path} from {request.remote_addr}")

@app.after_request
def after_request(response):
    if hasattr(g, 'start_time'):
        elapsed = time.time() - g.start_time
        response.headers['X-Response-Time'] = f"{elapsed:.3f}s"
        response.headers['X-Request-ID'] = g.request_id
    return response

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "status": 404,
        "request_id": g.get('request_id', 'unknown')
    }), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {str(error)}")
    return jsonify({
        "error": "Internal server error",
        "status": 500,
        "request_id": g.get('request_id', 'unknown')
    }), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({
        "error": "Rate limit exceeded",
        "message": str(e.description),
        "status": 429
    }), 429

# Validation function
def validate_input(data):
    """Validate input data against expected ranges and types"""
    errors = []
    
    # Check for missing features
    missing = [f for f in FEATURE_ORDER if f not in data]
    if missing:
        errors.append(f"Missing required features: {', '.join(missing)}")
        return errors
    
    # Validate each feature
    for feature, value in data.items():
        if feature not in FEATURE_VALIDATION:
            continue
            
        validation = FEATURE_VALIDATION[feature]
        
        # Type check
        if not isinstance(value, (int, float)):
            errors.append(f"{feature} must be a number")
            continue
        
        # Range check
        if value < validation["min"] or value > validation["max"]:
            errors.append(
                f"{feature} must be between {validation['min']} and {validation['max']}"
            )
    
    return errors

# Routes

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Oral Cancer Risk Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/health",
            "api_info": "/api/v1/info",
            "predict": "/api/v1/predict",
            "batch_predict": "/api/v1/predict/batch",
            "metadata": "/api/v1/metadata",
            "model_info": "/api/v1/model/info",
            "documentation": "/docs"
        },
        "timestamp": datetime.utcnow().isoformat()
    })

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint for monitoring"""
    model_status = "healthy" if model is not None else "unhealthy"
    overall_status = "healthy" if model_status == "healthy" else "degraded"
    
    return jsonify({
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "components": {
            "model": model_status,
            "api": "healthy"
        }
    })

@app.route("/api/v1/info", methods=["GET"])
def api_info():
    """General API information"""
    return jsonify({
        "api_name": "Oral Cancer Risk Prediction API",
        "version": "1.0.0",
        "description": "Machine learning API for predicting oral cancer risk based on pre-diagnosis features",
        "purpose": "Educational and demonstration purposes",
        "disclaimer": "⚠️ This API is for demonstration purposes only. Not for clinical use or medical diagnosis.",
        "author": "Godspower Ojoduene Amaha",
        "contact": "amahagodspower@gmail.com",
        "documentation": "http://oral-cancer.duckdns.org/docs",
        "github": "https://github.com/amaha2428/Oral-Cancer/",
        "rate_limits": {
            "per_hour": 50,
            "per_day": 200
        },
        "endpoints": [
            "/api/v1/predict",
            "/api/v1/predict/batch",
            "/api/v1/metadata",
            "/api/v1/model/info"
        ]
    })

@app.route("/api/v1/metadata", methods=["GET"])
def metadata():
    """Dataset and model metadata"""
    try:
        return jsonify({
            "model_type": "RandomForestClassifier",
            "task": "Binary classification (Oral Cancer Prediction)",
            "features_used": FEATURE_ORDER,
            "feature_count": len(FEATURE_ORDER),
            "target": "diagnosis",
            "target_labels": {
                "0": "No Oral Cancer",
                "1": "Oral Cancer"
            },
            "evaluation_metric": "F1 Score",
            "cv_f1_score": 0.4965,
            "dataset_size": 84922,
            "model_version": "1.0.0",
            "trained_on": "Pre-diagnosis self-reported symptoms and risk factors",
            "training_date": "2025-12-23",
            "framework": "scikit-learn"
        })
    except Exception as e:
        logger.error(f"Error in metadata endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/v1/model/info", methods=["GET"])
def model_info():
    """Detailed model information"""
    return jsonify({
        "model_name": "RandomForestClassifier",
        "version": "1.0.0",
        "trained_date": "2024-01-15",
        "framework": "scikit-learn",
        "algorithm": "Random Forest Ensemble",
        "performance": {
            "cv_f1_score": 0.4965,
            "cv_folds": 5,
            "train_test_split": "80/20",
            
        },
        "hyperparameters": {
            "note": "Default Random Forest parameters used for computational efficiency"
        },
        "input_features": len(FEATURE_ORDER),
        "output_classes": 2,
        "feature_importance_top_5": [
            {"feature": "age", "importance": 0.303},
            {"feature": "region", "importance": 0.114},
            {"feature": "diet_quality", "importance": 0.070},
            {"feature": "oral_lesions", "importance": 0.044},
            {"feature": "bleeding", "importance": 0.043}
        ]
    })

@app.route("/api/v1/predict", methods=["POST"])
@limiter.limit("20 per minute")
def predict():
    """Single prediction endpoint with validation"""
    try:
        if model is None:
            return jsonify({"error": "Model not available"}), 503
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No input data provided",
                "request_id": g.request_id
            }), 400
        
        # Validate input
        validation_errors = validate_input(data)
        if validation_errors:
            return jsonify({
                "error": "Input validation failed",
                "validation_errors": validation_errors,
                "request_id": g.request_id
            }), 400
        
        # Convert to DataFrame in correct order
        X = pd.DataFrame([[data[col] for col in FEATURE_ORDER]], columns=FEATURE_ORDER)
        
        # Make prediction
        prediction = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        
        response = {
            "request_id": g.request_id,
            "prediction": prediction,
            "prediction_label": "Oral Cancer" if prediction == 1 else "No Oral Cancer",
            "confidence": round(float(proba[prediction]), 4),
            "class_probabilities": {
                "no_cancer": round(float(proba[0]), 4),
                "oral_cancer": round(float(proba[1]), 4)
            },
            "risk_level": "High" if proba[1] > 0.7 else "Medium" if proba[1] > 0.4 else "Low",
            "timestamp": datetime.utcnow().isoformat(),
            "disclaimer": "This is a risk prediction, not a diagnosis. Consult a healthcare professional."
        }
        
        logger.info(f"Prediction completed: {g.request_id} - Result: {prediction}")
        return jsonify(response)
    
    except KeyError as e:
        logger.error(f"Missing feature in request {g.request_id}: {str(e)}")
        return jsonify({
            "error": f"Missing or invalid feature: {str(e)}",
            "request_id": g.request_id
        }), 400
    
    except Exception as e:
        logger.error(f"Error in prediction {g.request_id}: {str(e)}")
        return jsonify({
            "error": "Prediction failed",
            "details": str(e),
            "request_id": g.request_id
        }), 500

@app.route("/api/v1/predict/batch", methods=["POST"])
@limiter.limit("10 per minute")
def batch_predict():
    """Batch prediction endpoint"""
    try:
        if model is None:
            return jsonify({"error": "Model not available"}), 503
        
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({
                "error": "Input must be a list of patient records",
                "request_id": g.request_id
            }), 400
        
        if len(data) > 100:
            return jsonify({
                "error": "Batch size limited to 100 records",
                "request_id": g.request_id
            }), 400
        
        # Validate all records
        all_errors = []
        for idx, record in enumerate(data):
            errors = validate_input(record)
            if errors:
                all_errors.append({"record_index": idx, "errors": errors})
        
        if all_errors:
            return jsonify({
                "error": "Validation failed for some records",
                "validation_errors": all_errors,
                "request_id": g.request_id
            }), 400
        
        # Convert to DataFrame
        X = pd.DataFrame(data)
        
        # Reorder columns to match training
        X = X[FEATURE_ORDER]
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        results = []
        for i in range(len(predictions)):
            pred = int(predictions[i])
            proba = probabilities[i]
            results.append({
                "record_index": i,
                "prediction": pred,
                "prediction_label": "Oral Cancer" if pred == 1 else "No Oral Cancer",
                "confidence": round(float(proba[pred]), 4),
                "class_probabilities": {
                    "no_cancer": round(float(proba[0]), 4),
                    "oral_cancer": round(float(proba[1]), 4)
                },
                "risk_level": "High" if proba[1] > 0.7 else "Medium" if proba[1] > 0.4 else "Low"
            })
        
        logger.info(f"Batch prediction completed: {g.request_id} - Records: {len(results)}")
        
        return jsonify({
            "request_id": g.request_id,
            "num_records": len(results),
            "results": results,
            "timestamp": datetime.utcnow().isoformat(),
            "disclaimer": "These are risk predictions, not diagnoses. Consult healthcare professionals."
        })
    
    except Exception as e:
        logger.error(f"Error in batch prediction {g.request_id}: {str(e)}")
        return jsonify({
            "error": "Batch prediction failed",
            "details": str(e),
            "request_id": g.request_id
        }), 500

@app.route("/openapi.json")
def openapi():
    """Serve OpenAPI specification"""
    return send_from_directory(".", "openapi.json")

@app.route("/docs")
def docs():
    """Interactive API documentation"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Oral Cancer Prediction API Documentation</title>
        <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist/swagger-ui.css" />
        <style>
            body { margin: 0; padding: 0; }
            .topbar { display: none; }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://unpkg.com/swagger-ui-dist/swagger-ui-bundle.js"></script>
        <script>
        SwaggerUIBundle({
            url: '/openapi.json',
            dom_id: '#swagger-ui',
            deepLinking: true,
            presets: [
                SwaggerUIBundle.presets.apis,
                SwaggerUIBundle.SwaggerUIStandalonePreset
            ],
            layout: "BaseLayout"
        });
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5555)