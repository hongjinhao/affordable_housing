from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    PowerTransformer,
    StandardScaler,
)
from tqdm import tqdm
import typer

from affordable_housing.config import MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

TEST_SIZE = 0.25
SEED = 42


def binary_homeless(X):
    return (X > 0).astype(int)


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "merged_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR,
    model_path: Path = MODELS_DIR / "preprocesser.pkl",
    # -----------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    logger.info("Generating features from dataset...")

    # Read Excel files
    logger.info(f"Loading first Excel file from {input_path}")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded first dataset with {len(df)} rows and {len(df.columns)} columns")

    # extract important features
    logger.info("Extracting important columns from dataset")
    y_values = df["AWARD"].map({"Yes": 1, "No": 0})
    numeric = [
        "AVERAGE TARGETED AFFORDABILITY",
        "CDLAC TOTAL POINTS SCORE",
        "CDLAC TIE-BREAKER SELF SCORE",
        "BOND REQUEST",
        "HOMELESS %",
    ]
    cat = [
        "CONSTRUCTION TYPE",
        "HOUSING TYPE",
        "CDLAC POOL",
        "NEW CONSTRUCTION SET ASIDE",
        "CDLAC REGION",
    ]
    X_values = df[numeric + cat]
    logger.info(
        f"Extracted {X_values.shape} and {y_values.shape} important features from the dataset"
    )

    # rename column names to single word and lowercase
    rename_dict = {
        "AVERAGE TARGETED AFFORDABILITY": "avg_targeted_affordability",
        "CDLAC TOTAL POINTS SCORE": "CDLAC_total_points_score",
        "CDLAC TIE-BREAKER SELF SCORE": "CDLAC_tie_breaker_self_score",
        "BOND REQUEST": "bond_request_amount",
        "HOMELESS %": "homeless_percent",
        "CONSTRUCTION TYPE": "construction_type",
        "HOUSING TYPE": "housing_type",
        "CDLAC POOL": "CDLAC_pool_type",
        "NEW CONSTRUCTION SET ASIDE": "new_construction_set_aside",
        "CDLAC REGION": "CDLAC_region",
    }
    X_values = X_values.rename(columns=rename_dict)
    print(X_values.columns)
    # split
    logger.info("Split data into test and train")
    X_train, X_test, y_train, y_test = train_test_split(
        X_values, y_values, test_size=TEST_SIZE, stratify=y_values, random_state=SEED
    )
    logger.info(f"split completed with test size of {TEST_SIZE}")

    # pipelines
    logger.info("Creating preprocessing pipelines")

    logger.debug("Setting up homeless pipeline")
    homeless_transformer = FunctionTransformer(
        func=binary_homeless,
        feature_names_out="one-to-one",
    )
    homeless_pipe = make_pipeline(homeless_transformer)

    logger.debug("Setting up points pipeline")
    points_transformer = PowerTransformer(method="yeo-johnson")
    points_pipe = make_pipeline(points_transformer, MinMaxScaler())

    logger.debug("Setting up categorical and numerical pipelines")
    cat_pipe = make_pipeline(OneHotEncoder())
    remainder_num_pipe = make_pipeline(StandardScaler())

    logger.info("Creating column transformer")
    renamed_cat = [
        "construction_type",
        "housing_type",
        "CDLAC_pool_type",
        "new_construction_set_aside",
        "CDLAC_region",
    ]

    preprocessor_pipe = ColumnTransformer(
        transformers=[
            ("homeless_binary", homeless_pipe, ["homeless_percent"]),
            ("points_power", points_pipe, ["CDLAC_total_points_score"]),
            ("category", cat_pipe, renamed_cat),
        ],
        remainder=remainder_num_pipe,
    )
    # transform
    logger.info("Fitting and transforming training data")
    X_train_transform = preprocessor_pipe.fit_transform(X_train)
    logger.info("Transforming test data")
    X_test_transform = preprocessor_pipe.transform(X_test)

    # Convert to DataFrame with column names
    logger.info("Converting transformed data to DataFrame")
    X_train_transform_df = pd.DataFrame(
        X_train_transform, columns=preprocessor_pipe.get_feature_names_out()
    )
    X_test_transform_df = pd.DataFrame(
        X_test_transform, columns=preprocessor_pipe.get_feature_names_out()
    )
    # save features
    logger.info(f"Saving features to {output_path}")
    X_train.to_csv(output_path / "X_train.csv", index=False)
    X_test.to_csv(output_path / "X_test.csv", index=False)
    X_train_transform_df.to_csv(output_path / "X_train_transform.csv", index=False)
    X_test_transform_df.to_csv(output_path / "X_test_transform.csv", index=False)
    y_train.to_csv(output_path / "y_train.csv", index=False)
    y_test.to_csv(output_path / "y_test.csv", index=False)
    logger.info("Features saved successfully.")

    # save preprocessor pipeline
    logger.info("Saving preprocessor pipeline")
    joblib.dump(preprocessor_pipe, model_path)
    logger.info(f"Preprocessor pipeline saved to {model_path}")

    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
