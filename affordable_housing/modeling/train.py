from pathlib import Path

import joblib
from loguru import logger
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
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
)
import typer

from affordable_housing.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def binary_homeless(X):
    """Convert homeless percentage to binary (1 if > 0, else 0)."""
    return (X > 0).astype(int)


def get_binary_homeless_transformer():
    """Return a FunctionTransformer for binary_homeless."""
    return FunctionTransformer(
        func=binary_homeless,
        feature_names_out="one-to-one",
    )


def f1_scores_comparision(y_train, y_train_pred, model, X):
    train_f1 = f1_score(y_train, y_train_pred)
    print(f"Train             F1 score: {train_f1:.3f}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y_train, cv=cv, scoring="f1")
    print(f"Cross-validation F1 scores: {scores.mean():.3f} ± {scores.std():.3f}")


def run_random_search_cv(model, param_dist, X, y, n_iter=20, scoring="f1", cv=5, random_state=42):
    """
    Run RandomizedSearchCV for a given model and parameter distribution.
    Prints best estimator parameters, metrics, and uses f1_scores_comparision.
    """
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
    )
    search.fit(X, y)
    print(f"Best Validation {scoring.upper()} (CV): {search.best_score_:.3f}")
    print("Best parameters:", search.best_params_)
    best_model = search.best_estimator_
    y_pred = best_model.predict(X)
    print("model predictions on train: ", y_pred[:20])
    print("actual y values:            ", y[:20])
    print(classification_report(y, y_pred))
    f1_scores_comparision(y, y_pred, best_model, X)
    return best_model, search.best_params_, search


@app.command()
def main(
    dataset_path: Path = PROCESSED_DATA_DIR / "3yr_dataset_train.csv",
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
    logger.info(f"Saving best model to {model_path}")
    joblib.dump(best_model, model_path)
    logger.success("Model training and saving complete.")


if __name__ == "__main__":
    app()
