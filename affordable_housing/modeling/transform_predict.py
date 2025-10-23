from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report, f1_score
import typer

from affordable_housing.config import (
    EXTERNAL_DATA_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
)
from affordable_housing.dataset import (
    clean_and_merge_columns,
    clean_construction_type,
    clean_region,
    rename_column_names,
    standardize_application_number,
)
from affordable_housing.logger_config import setup_debug_logger

app = typer.Typer()

logger, log_file = setup_debug_logger()


def transform_new_construction_set_aside(df_round2: pd.DataFrame) -> pd.DataFrame:
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


def prepare_round2_for_model(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare raw Round2 applicant list into the feature columns used by the trained model.
    Returns a DataFrame with columns:
      ['application_number','avg_targeted_affordability','total_points',
       'tie_breaker_self_score','bond_request_amount','num_homeless_units',
       'construction_type','housing_type','CDLAC_region',
       'combined_CDLAC_pool','combined_set_aside']
    Missing numeric values are coerced to 0, categorical to 'NONE'.
    """
    df = raw_df.copy()

    # Create NEW CONSTRUCTION SET ASIDE (expects HOMELESS, ELI/VLI, MIP in raw)
    df = transform_new_construction_set_aside(df)

    # Normalize column names to your standard names
    df = rename_column_names(df)

    # Merge/clean set-asides and CDLAC pool etc.
    df = clean_and_merge_columns(df)

    # Standardize application number if present
    if "application_number" in df.columns:
        df["application_number"] = (
            df["application_number"].astype(str).apply(standardize_application_number)
        )

    # Clean region and construction type
    if "CDLAC_region" in df.columns:
        df["CDLAC_region"] = df["CDLAC_region"].fillna("NONE").astype(str).apply(clean_region)
    if "construction_type" in df.columns:
        df["construction_type"] = (
            df["construction_type"].fillna("NONE").astype(str).apply(clean_construction_type)
        )

    # Ensure combined_set_aside and combined_CDLAC_pool exist
    df["combined_set_aside"] = df.get("combined_set_aside", pd.Series(index=df.index)).fillna(
        "NONE"
    )
    df["combined_CDLAC_pool"] = df.get("combined_CDLAC_pool", pd.Series(index=df.index)).fillna(
        "NONE"
    )

    # Numeric columns: coerce and fill with 0
    num_cols = {
        "avg_targeted_affordability": [
            "AVERAGE TARGETED AFFORDABILITY",
            "avg_targeted_affordability",
        ],
        "total_points": ["CDLAC TOTAL POINTS", "total_points"],
        "tie_breaker_self_score": ["TIEBREAKER SELF SCORE", "tie_breaker_self_score"],
        "bond_request_amount": ["BOND REQUEST", "bond_request_amount"],
        "num_homeless_units": ["units for homeless", "num_homeless_units", "HOMELESS %"],
    }
    # try to pick existing column names (some files use different raw headers)
    for canonical, possibles in num_cols.items():
        for p in possibles:
            if p in df.columns:
                df[canonical] = pd.to_numeric(df[p], errors="coerce")
                break
        df[canonical] = df.get(canonical, pd.Series(index=df.index)).fillna(0)

    # Ensure housing_type exists
    df["housing_type"] = df.get("housing_type", pd.Series(index=df.index)).fillna("NONE")

    # Final column order for model
    final_cols = [
        "application_number",
        "avg_targeted_affordability",
        "total_points",
        "tie_breaker_self_score",
        "bond_request_amount",
        "num_homeless_units",
        "construction_type",
        "housing_type",
        "CDLAC_region",
        "combined_CDLAC_pool",
        "combined_set_aside",
    ]
    prepared = df.reindex(columns=final_cols)

    # Cast strings to str and strip where appropriate
    for c in [
        "construction_type",
        "housing_type",
        "combined_CDLAC_pool",
        "combined_set_aside",
        "CDLAC_region",
    ]:
        if c in prepared.columns:
            prepared[c] = prepared[c].astype(str).str.strip()

    return prepared


@app.command()
def main(
    input_path: Path = EXTERNAL_DATA_DIR / "2025-R2-ApplicantList.xlsx",
    preprocessor_path: Path = MODELS_DIR / "3yr-preprocessor.pkl",
    model_path: Path = MODELS_DIR / "3yr-model.pkl",
    output_path: Path = PROCESSED_DATA_DIR / "predictions/2025-R2-predictions-3yrmodel2.csv",
    manual_award_path: Path = INTERIM_DATA_DIR / "2025-R2-manual-award.csv",
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

        # Load preprocessor and model
        model = joblib.load(model_path)

        prepared_X = prepare_round2_for_model(raw_df)
        prepared_X = prepared_X.drop(columns=["application_number"])
        # Log shape and columns first
        logger.debug(f"Prepared data shape: {prepared_X.shape}")
        logger.debug(f"Columns: {list(prepared_X.columns)}")

        # Then show the actual data with full display
        with pd.option_context("display.max_columns", None, "display.width", None):
            logger.debug(f"Sample data:\n{prepared_X.head()}")

        # Generate predictions
        logger.info("Performing inference...")
        y_pred = model.predict(prepared_X)
        logger.info(f"First 20 predictions: {y_pred[:20]}")

        # If manual awards provided, evaluate performance
        if manual_award_path.exists():
            logger.info(f"Loading manual awards from {manual_award_path}")
            manual_df = pd.read_csv(manual_award_path)
            manual_df["AWARD"] = manual_df["AWARD"].map({"Yes": 1, "No": 0})
            manual_df = rename_column_names(manual_df)
            manual_df["application_number"] = (
                manual_df["application_number"].astype(str).apply(standardize_application_number)
            )
            merged = prepared_X.merge(
                manual_df[["application_number", "AWARD"]],
                left_index=True,
                right_index=True,
                how="left",
            )
            y_true = merged["AWARD"]
            logger.info("Classification Report against manual awards:")
            logger.info("\n" + classification_report(y_true, y_pred, target_names=["No", "Yes"]))
            f1 = f1_score(y_true, y_pred)
            logger.info(f"F1 Score: {f1:.4f}")
        # Create output DataFrame with raw data and predictions
        logger.info("Merging raw data with predictions")
        output_df = raw_df.copy()
        output_df["PREDICTED_AWARD"] = y_pred
        output_df["PREDICTED_AWARD"] = output_df["PREDICTED_AWARD"].map({1: "Yes", 0: "No"})

        # Save merged dataset
        output_df.to_csv(output_path, index=False)
        logger.success(f"Processing complete. Saved to {output_path}")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise


if __name__ == "__main__":
    app()
