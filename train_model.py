import os
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np
from mlflow.models.signature import infer_signature

def train_model():
    # Load dataset
    data = load_iris()
    X = data.data
    y = data.target

    # Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Start MLflow run
    mlflow.start_run()
    
    # Log the accuracy
    mlflow.log_metric("accuracy", accuracy)
    
    # Example input data (one row of features)
    input_example = pd.DataFrame([X_train[0]], columns=data.feature_names)

    # Infer the model signature based on input data and the trained model
    signature = infer_signature(X_train, model.predict(X_train))

    # Set the artifact location to a valid path
    artifact_location = "./mlflow_artifacts"

    # Create the directory if it doesn't exist
    if not os.path.exists(artifact_location):
        os.makedirs(artifact_location)
    
    # Log the model with mlflow
    mlflow.set_artifact_uri(artifact_location)

    # Log the model with signature and input example
    mlflow.sklearn.log_model(model, "models/model", signature=signature, input_example=input_example)
    
    # Save the model locally for deployment
    save_path = "models/model"
    os.makedirs("models", exist_ok=True)  # Ensure the directory exists
    mlflow.sklearn.save_model(model, save_path)
    print(f"Model saved locally to: {save_path}")
    
    mlflow.end_run()

if __name__ == "__main__":
    train_model()