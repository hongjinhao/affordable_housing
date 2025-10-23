from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
)
import typer

from affordable_housing.config import MODELS_DIR, PROCESSED_DATA_DIR
from affordable_housing.logger_config import setup_training_logger
from affordable_housing.utils import get_binary_homeless_transformer

app = typer.Typer()

logger, log_file = setup_training_logger()


def evaluate_model_performance(X, y, model, dataset="train", include_cv=True):
    """
    Evaluate model performance with predictions, F1 score, and classification report.

    Args:
        X: Feature matrix
        y: True labels
        model: Fitted model to evaluate
        dataset: Name of dataset being evaluated (e.g., "train", "test", "validation")
        include_cv: Whether to include cross-validation (default: True, set False for test set)

    Returns:
        dict: Dictionary containing f1_score, predictions, and optionally cv_scores
    """
    logger.info(f"Evaluating model performance on {dataset} set...")

    # Make predictions
    y_pred = model.predict(X)
    logger.debug(f"Model predictions (first 20): {y_pred[:20]}")
    logger.debug(f"Actual y values (first 20): {y[:20]}")

    # Calculate F1 score
    f1 = f1_score(y, y_pred)
    logger.info(f"{dataset.capitalize()} F1 score: {f1:.3f}")

    # Classification report
    report = classification_report(y, y_pred)
    logger.info(f"{dataset.capitalize()} Classification Report:\n{report}")

    results = {"f1_score": f1, "predictions": y_pred, "classification_report": report}

    # Cross-validation only if requested (typically for training set)
    if include_cv:
        logger.info("Performing cross-validation...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring="f1")
        logger.info(f"Cross-validation F1 scores: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        logger.debug(f"Individual CV fold scores: {cv_scores}")
        results["cv_mean"] = cv_scores.mean()
        results["cv_std"] = cv_scores.std()

    return results


def run_random_search_cv(model, param_dist, X, y, n_iter=20, scoring="f1", cv=5, random_state=42):
    """
    Run RandomizedSearchCV for a given model and parameter distribution.
    Prints best estimator parameters, metrics, and uses f1_scores_comparision.
    """
    logger.info("Starting RandomizedSearchCV...")
    logger.info(f"Parameter search space: {param_dist}")
    logger.info(f"Search iterations: {n_iter}, CV folds: {cv}, Scoring: {scoring}")

    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        verbose=1,
    )

    logger.info("Fitting RandomizedSearchCV...")
    search.fit(X, y)

    logger.info(f"Best Validation {scoring.upper()} (CV): {search.best_score_:.3f}")
    logger.info(f"Best parameters: {search.best_params_}")

    best_model = search.best_estimator_
    evaluate_model_performance(X, y, best_model, "Train")
    return best_model, search.best_params_, search


@app.command()
def main(
    dataset_path: Path = PROCESSED_DATA_DIR / "3yr_dataset_train.csv",
    test_dataset_path: Path = PROCESSED_DATA_DIR / "3yr_dataset_test.csv",
    model_path: Path = MODELS_DIR / "3yr-model.pkl",
):
    logger.info("Loading training data...")
    df = pd.read_csv(dataset_path)
    X_train = df.drop(columns=["award", "application_number"])
    y_train = df["award"].map({"Yes": 1, "No": 0}).values.ravel()
    logger.info(f"Training data has {X_train.shape[0]} rows and {X_train.shape[1]} features.")
    logger.info(f"Training label shape: {y_train.shape}")

    logger.info("Setting up preprocessor + model pipeline...")
    homeless_pipe = make_pipeline(get_binary_homeless_transformer())
    points_transformer = PowerTransformer(method="yeo-johnson")
    points_pipe = make_pipeline(points_transformer, MinMaxScaler())
    cat_pipe = make_pipeline(OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    remainder_num_pipe = make_pipeline(StandardScaler())

    categorical_names = [
        "construction_type",
        "housing_type",
        "combined_CDLAC_pool",
        "combined_set_aside",
        "CDLAC_region",
    ]

    preprocessor_pipe = ColumnTransformer(
        transformers=[
            ("homeless_binary", homeless_pipe, ["num_homeless_units"]),
            ("points_power", points_pipe, ["total_points"]),
            ("category", cat_pipe, categorical_names),
        ],
        remainder=remainder_num_pipe,
    )

    full_pipeline = make_pipeline(preprocessor_pipe, RandomForestClassifier(random_state=42))

    param_dist = {
        "randomforestclassifier__n_estimators": randint(50, 201),
        "randomforestclassifier__max_depth": [None] + list(range(1, 21)),
        "randomforestclassifier__min_samples_split": randint(2, 11),
    }
    logger.info("Starting hyperparameter tuning with RandomizedSearchCV...")
    best_model, _, _ = run_random_search_cv(
        full_pipeline, param_dist, X_train, y_train, n_iter=10, cv=3
    )

    X_test = pd.read_csv(test_dataset_path).drop(columns=["award", "application_number"])
    y_test = pd.read_csv(test_dataset_path)["award"].map({"Yes": 1, "No": 0}).values.ravel()
    evaluate_model_performance(X_test, y_test, best_model, "test", include_cv=False)

    logger.info(f"Saving best model to {model_path}")
    joblib.dump(best_model, model_path)
    logger.success("Model training and saving complete.")


if __name__ == "__main__":
    app()
