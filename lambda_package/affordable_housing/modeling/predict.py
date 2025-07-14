from pathlib import Path

from loguru import logger
import pandas as pd
import typer

from affordable_housing.config import MODELS_DIR, PROCESSED_DATA_DIR
import joblib
from sklearn.metrics import classification_report, f1_score

app = typer.Typer()


def predict(
    user_input: pd.DataFrame,
    model_path: Path = MODELS_DIR / "model.pkl",
    preprocessor_path: Path = MODELS_DIR / "preprocessor.pkl",
) -> pd.Series:
    """Perform inference on input features using the specified model. Use for API endpoint.

    Args:
        model_path (Path): Path to the trained model (.pkl file).
        features (pd.DataFrame): Input features for prediction.

    Returns:
        dictionary: Predicted labels and probability
    """
    logger.info("Loading model...")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    logger.info("Preprocessing input...")
    transformed_features = preprocessor.transform(user_input)
    logger.info("Performing inference...")
    prediction = model.predict(transformed_features)
    prob = model.predict_proba(transformed_features)[:, 1][0]
    logger.info(f"prediction: {prediction}")
    logger.info(f"probability: {prob}")

    return {"prediction": prediction, "probability": prob}


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "X_test_transform.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    y_test_path: Path = PROCESSED_DATA_DIR / "y_test.csv",
    # -----------------------------------------
):
    logger.info("Loading test features and model...")
    X_test = pd.read_csv(features_path)
    model = joblib.load(model_path)

    logger.info("Performing inference...")
    y_test_pred = model.predict(X_test)
    logger.info(f"First 20 predictions: {y_test_pred[:20]}")

    # Optionally compare to actual y_test if available
    if y_test_path.exists():
        y_test = pd.read_csv(y_test_path).squeeze()
        logger.info(f"First 20 actual values: {y_test[:20].values}")
        f1 = f1_score(y_test, y_test_pred)
        logger.info(f"Test F1 score: {f1:.3f}")
        logger.info("Classification report:\n" + classification_report(y_test, y_test_pred))

    # Save predictions
    pd.Series(y_test_pred).to_csv(predictions_path, index=False)
    logger.success(f"Inference complete. Predictions saved to {predictions_path}")


if __name__ == "__main__":
    app()
