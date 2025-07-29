from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import typer

from affordable_housing.config import EXTERNAL_DATA_DIR, MODELS_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def transform_new_construction_set_aside(df_round2):
    """
    Transform Round 2 Homeless, ELI/VLI, and MIP columns into a single NEW CONSTRUCTION SET ASIDE column
    to match Round 1 format.

    Args:
        df_round2 (pd.DataFrame): Round 2 dataset with Homeless, ELI/VLI, and MIP columns.

    Returns:
        pd.DataFrame: Transformed DataFrame with NEW CONSTRUCTION SET ASIDE column and original columns dropped.
    """
    logger.info("Transforming NEW CONSTRUCTION SET ASIDE for Round 2 data...")

    # Copy the DataFrame to avoid modifying the original
    df = df_round2.copy()

    # Log unique values for debugging
    for col in ["HOMELESS", "ELI/VLI", "MIP"]:
        if col in df.columns:
            logger.info(f"{col} values: {df[col].unique()}")
        else:
            logger.error(f"Column {col} not found in Round 2 dataset")
            raise ValueError(f"Missing column {col}")

    # Convert YES/NO to 1/0 if necessary
    for col in ["HOMELESS", "ELI/VLI", "MIP"]:
        if df[col].dtype == "object":
            df[col] = df[col].replace({"Yes": 1, "No": 0}).fillna(0).astype(int)
        elif df[col].isna().any():
            logger.warning(f"Found missing values in {col}, imputing with 0")
            df[col] = df[col].fillna(0).astype(int)

    # Create NEW CONSTRUCTION SET ASIDE column
    def map_set_aside(row):
        if row["HOMELESS"] == 1 and row["ELI/VLI"] == 1:
            return "Homeless, ELI/VLI"
        elif row["ELI/VLI"] == 1:
            return "ELI/VLI"
        else:
            return "none"  # MIP = 1 or all 0s map to 'none'

    df["NEW CONSTRUCTION SET ASIDE"] = df.apply(map_set_aside, axis=1)

    # Verify valid categories
    valid_categories = ["none", "Homeless, ELI/VLI", "ELI/VLI"]
    invalid_categories = df[~df["NEW CONSTRUCTION SET ASIDE"].isin(valid_categories)][
        "NEW CONSTRUCTION SET ASIDE"
    ].unique()
    if len(invalid_categories) > 0:
        logger.error(f"Invalid NEW CONSTRUCTION SET ASIDE values: {invalid_categories}")
        raise ValueError("Invalid NEW CONSTRUCTION SET ASIDE values detected")

    # Drop original columns
    df = df.drop(columns=["HOMELESS", "ELI/VLI", "MIP"])
    logger.info("Dropped Homeless, ELI/VLI, and MIP columns")

    logger.info("NEW CONSTRUCTION SET ASIDE transformation complete")
    return df


@app.command()
def main(
    input_path: Path = EXTERNAL_DATA_DIR / "2025-R2-ApplicantList.xlsx",
    preprocessor_path: Path = MODELS_DIR / "preprocessor.pkl",
    model_path: Path = MODELS_DIR / "model.pkl",
    output_path: Path = PROCESSED_DATA_DIR / "2025-R2-predictions-with-raw.csv",
    decision_threshold: float = 0.44,  # Lowered to reflect OBBBA's increased LIHTC
):
    """
    Transform raw data using the preprocessor, generate predictions using the model,
    and save a merged dataset containing raw data and predictions.
    """
    logger.info("Starting prediction with transformation...")

    try:
        # Load raw data
        logger.info(f"Loading raw dataset from {input_path}")
        raw_df = pd.read_excel(input_path, header=1, index_col=None)
        logger.info(f"Loaded dataset with {len(raw_df)} rows and {len(raw_df.columns)} columns")

        # Create new column "NEW CONSTRUCTION SET ASIDE"
        logger.info("Creating new column to match round 1")
        raw_df = transform_new_construction_set_aside(raw_df)

        # modify the column "CONSTRUCTION TYPE"
        raw_df["CONSTRUCTION TYPE"] = raw_df["CONSTRUCTION TYPE"].replace(
            {"Acquisition/Rehabilitation": "Acq and Rehabilitation"}
        )

        # Extract features for transformation
        logger.info("Extracting features for transformation")
        numeric = [
            "AVERAGE TARGETED AFFORDABILITY",
            "CDLAC TOTAL POINTS",
            "TIEBREAKER SELF SCORE",
            "BOND REQUEST",
            "HOMELESS %",
        ]
        cat = [
            "CONSTRUCTION TYPE",
            "HOUSING TYPE",
            "CDLAC POOL",
            "NEW CONSTRUCTION SET ASIDE",  # Newly created column
            "CDLAC REGION",
        ]
        X_values = raw_df[numeric + cat]
        logger.info("Feature extraction complete")

        # rename column names to single word and lowercase
        logger.info("rename columns")
        rename_dict = {
            "AVERAGE TARGETED AFFORDABILITY": "avg_targeted_affordability",
            "CDLAC TOTAL POINTS": "CDLAC_total_points_score",
            "TIEBREAKER SELF SCORE": "CDLAC_tie_breaker_self_score",
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
        logger.info("rename columns successful!")

        # Load preprocessor and model
        logger.info(f"Loading preprocessor from {preprocessor_path}")
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Loading model from {model_path}")
        model = joblib.load(model_path)

        # Transform features
        logger.info("Transforming features...")
        X_transformed = preprocessor.transform(X_values)
        X_transformed_df = pd.DataFrame(
            X_transformed, columns=preprocessor.get_feature_names_out()
        )
        logger.info("Feature transformation complete")

        # Generate predictions
        logger.info("Performing inference...")
        y_pred_proba = model.predict_proba(X_transformed)[:, 1]
        y_pred = (y_pred_proba >= decision_threshold).astype(int)
        logger.info(f"First 20 predictions: {y_pred[:20]}")

        # Optionally compare to actual labels if available
        if "AWARD" in raw_df.columns:
            y_true = raw_df["AWARD"].map({"Yes": 1, "No": 0})
            logger.info(f"First 20 actual values: {y_true[:20].values}")
            f1 = f1_score(y_true, y_pred)
            logger.info(f"F1 score: {f1:.3f}")
            logger.info("Classification report:\n" + classification_report(y_true, y_pred))

        # Create output DataFrame with raw data and predictions
        logger.info("Merging raw data with predictions")
        output_df = raw_df.copy()
        output_df["PREDICTED_AWARD"] = y_pred
        output_df["PREDICTION_PROBABILITY"] = y_pred_proba

        # Map numeric predictions back to Yes/No for readability
        output_df["PREDICTED_AWARD"] = output_df["PREDICTED_AWARD"].map({1: "Yes", 0: "No"})

        # Save merged dataset
        logger.info(f"Saving merged dataset with predictions to {output_path}")
        output_df.to_csv(output_path, index=False)
        logger.success(f"Processing complete. Saved to {output_path}")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    app()
