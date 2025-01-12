import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
import numpy as np
from loguru import logger
from prometheus_client import Counter, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST
from flask import Response

# Initialize Flask app
app = Flask(__name__)

# Logging configuration
logger.add("logs/model_deployment.log", rotation="1 day")
logger.info("Model API initialized")

# Prometheus monitoring
REQUEST_COUNT = Counter('request_count', 'Total count of requests to the API')
PREDICTION_COUNT = Counter('prediction_count', 'Total count of predictions made by the API')

# Load the trained model
logger.info("Loading trained model...")
model = mlflow.sklearn.load_model("models/model")
logger.info("Model loaded successfully")

# Prometheus metrics endpoint
@app.route('/metrics', methods=['GET'])
def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

@app.before_request
def before_request():
    """Increment request count before handling a request."""
    REQUEST_COUNT.inc()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        logger.info(f"Received prediction request: {data}")
        
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features)
        
        PREDICTION_COUNT.inc()  # Increment prediction count metric
        logger.info(f"Prediction made: {prediction.tolist()}")
        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(debug=True)
