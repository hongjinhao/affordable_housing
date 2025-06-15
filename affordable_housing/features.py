from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from tqdm import tqdm
import typer

from affordable_housing.config import PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "merged_dataset.csv",
    output_path: Path = PROCESSED_DATA_DIR,
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
    logger.info("Extraction complete")

    # split
    logger.info("Split data into test and train")
    X_train, X_test, y_train, y_test = train_test_split(
        X_values, y_values, test_size=0.25, stratify=y_values, random_state=42
    )
    logger.info("split data into test and train")

    # transform
    X_train["HOMELESS_present"] = (X_train["HOMELESS %"] > 0).astype(int)
    numeric.remove("HOMELESS %")
    cat.append("HOMELESS_present")
    X_train.drop("HOMELESS %")

    # save features
    logger.info(f"Saving features to {output_path}")
    X_train.to_csv(output_path / "X_train.csv", index=False)
    X_test.to_csv(output_path / "X_test.csv", index=False)
    y_train.to_csv(output_path / "y_train.csv", index=False)
    y_test.to_csv(output_path / "y_test.csv", index=False)
    logger.info("Features saved successfully.")

    logger.success("Features generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
