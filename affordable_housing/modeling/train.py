from pathlib import Path

import joblib
from loguru import logger
import mlflow
import pandas as pd
from scipy.stats import uniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
import typer

from affordable_housing.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    features_path: Path = PROCESSED_DATA_DIR / "X_train_transform.csv",
    labels_path: Path = PROCESSED_DATA_DIR / "y_train.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
):
    logger.info("Loading training data...")
    X_train = pd.read_csv(features_path)
    y_train = pd.read_csv(labels_path).squeeze()

    logger.info("Setting up model pipeline and hyperparameter search...")

    # Since we have already transformed features, no need for preprocessing
    full_pipeline = make_pipeline(LogisticRegression(random_state=42))

    param_dist = {
        "logisticregression__C": uniform(0.001, 1000),
        "logisticregression__penalty": ["l1", "l2"],
        "logisticregression__solver": ["liblinear", "saga", "lbfgs"],
        "logisticregression__class_weight": [None, "balanced"],
        "logisticregression__max_iter": [100, 200, 500],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_dist,
        n_iter=50,
        cv=cv,
        scoring="f1",
        random_state=42,
        verbose=1,
    )
    mlflow.set_experiment("AffordableHousing")
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name="2025R1Train"):
        logger.info("Fitting model...")
        random_search.fit(X_train, y_train)
        logger.info(f"Best Validation F1 (CV): {random_search.best_score_:.3f}")
        logger.info(f"Best Parameters: {random_search.best_params_}")

        best_model_pipeline = random_search.best_estimator_

        logger.info(f"Saving best model to {model_path}")
        joblib.dump(best_model_pipeline, model_path)
        logger.success("Model training and saving complete.")


if __name__ == "__main__":
    app()
