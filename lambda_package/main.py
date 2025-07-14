import json

import joblib
import pandas as pd

HEADER_CORS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",  # Add CORS header
    "Access-Control-Allow-Methods": "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-Amz-Date, Authorization, X-Api-Key, X-Amz-Security-Token",
}


def predict(
    user_input: dict,
    model_path: str = "models/model.pkl",
    preprocessor_path: str = "models/preprocessor.pkl",
) -> dict:
    """Perform inference on input features using the specified model.

    Args:
        user_input (dict): Input features for prediction.
        model_path (str): Path to the trained model (.pkl file).
        preprocessor_path (str): Path to the preprocessor (.pkl file).

    Returns:
        dict: Predicted labels and probability
    """
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    # Convert dict to list of values for sklearn
    features = pd.DataFrame([user_input])
    transformed_features = preprocessor.transform(features)
    prediction = model.predict(transformed_features)
    prob = model.predict_proba(transformed_features)[:, 1][0]

    return {"prediction": int(prediction), "probability": float(prob)}


def lambda_handler(event, context):
    """AWS Lambda handler for housing prediction API."""
    try:
        if "resource" not in event:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No resource provided in request"}),
                "headers": HEADER_CORS,
            }

        endpoint = event["resource"]

        if endpoint == "/health":
            return {
                "statusCode": 200,
                "body": json.dumps({"status": "Healthy!"}),
                "headers": HEADER_CORS,
            }
        elif endpoint == "/predict":
            # Proceed with prediction logic
            pass
        else:
            return {
                "statusCode": 404,
                "body": json.dumps({"error": f"Unsupported endpoint: {endpoint}"}),
                "headers": HEADER_CORS,
            }

        # Parse input from API Gateway
        if "body" not in event:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "No body provided in request"}),
                "headers": HEADER_CORS,
            }

        body = event["body"]
        if isinstance(body, str):
            body = json.loads(body)

        # Validate required fields
        required_fields = [
            "avg_targeted_affordability",
            "CDLAC_total_points_score",
            "CDLAC_tie_breaker_self_score",
            "bond_request_amount",
            "homeless_percent",
            "construction_type",
            "housing_type",
            "CDLAC_pool_type",
            "new_construction_set_aside",
            "CDLAC_region",
        ]
        if not all(field in body for field in required_fields):
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing required input fields"}),
                "headers": HEADER_CORS,
            }

        # Validate field types
        try:
            input_data = {
                "avg_targeted_affordability": float(body["avg_targeted_affordability"]),
                "CDLAC_total_points_score": int(body["CDLAC_total_points_score"]),
                "CDLAC_tie_breaker_self_score": float(body["CDLAC_tie_breaker_self_score"]),
                "bond_request_amount": float(body["bond_request_amount"]),
                "homeless_percent": float(body["homeless_percent"]),
                "construction_type": str(body["construction_type"]),
                "housing_type": str(body["housing_type"]),
                "CDLAC_pool_type": str(body["CDLAC_pool_type"]),
                "new_construction_set_aside": str(body["new_construction_set_aside"]),
                "CDLAC_region": str(body["CDLAC_region"]),
            }
        except (ValueError, TypeError):
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Invalid input data types"}),
                "headers": HEADER_CORS,
            }

        # Check model and preprocessor files
        model_path = "models/model.pkl"
        preprocessor_path = "models/preprocessor.pkl"
        try:
            with open(model_path, "rb"):
                pass
            with open(preprocessor_path, "rb"):
                pass
        except FileNotFoundError:
            return {
                "statusCode": 500,
                "body": json.dumps({"error": "Model or preprocessor file not found"}),
                "headers": HEADER_CORS,
            }

        # Perform prediction
        result = predict(input_data, model_path, preprocessor_path)

        # Return response
        return {"statusCode": 200, "body": json.dumps(result), "headers": HEADER_CORS}

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)}), "headers": HEADER_CORS}
