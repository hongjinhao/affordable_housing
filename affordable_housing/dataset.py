from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from affordable_housing.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def hello():
    print("Hello")


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path1: Path = EXTERNAL_DATA_DIR / "award_list.xlsx",
    input_path2: Path = EXTERNAL_DATA_DIR / "2025-Applicant-list-4-per-R1.xlsx",
    output_path: Path = PROCESSED_DATA_DIR / "merged_dataset.csv",
    merge_key: str = "APPLICATION NUMBER",  # Default merge key; adjust as needed
    merge_how: str = "right",  # Default merge type; options: left, right, inner, outer
    # ----------------------------------------------
):
    # ---- REPLACE THIS WITH YOUR OWN CODE ----
    """Process two Excel files by merging them and handling NaN values."""
    logger.info("Starting dataset processing...")

    try:
        # Read Excel files
        logger.info(f"Loading first Excel file from {input_path1}")
        df1 = pd.read_excel(input_path1)
        logger.info(f"Loaded first dataset with {len(df1)} rows and {len(df1.columns)} columns")

        logger.info(f"Loading second Excel file from {input_path2}")
        # ignore first row of the applicant data table
        df2 = pd.read_excel(input_path2, header=1, index_col=None)
        logger.info(f"Loaded second dataset with {len(df2)} rows and {len(df2.columns)} columns")

        logger.info(f"Extracting important columns from {input_path1}")
        important_columns = [
            "APPLICATION NUMBER",
            "NC POOL SELECTION: HOMELESS, ELI/VLI, MIP",
            "AWARD",
        ]
        df1_important = df1[important_columns]
        logger.info(f"Important columns are {important_columns}")

        # Merge DataFrames
        logger.info(f"Merging datasets on key '{merge_key}' with {merge_how} join")
        merged_df = pd.merge(df1_important, df2, on=merge_key, how=merge_how)
        logger.info(
            f"Merged dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns"
        )

        # Adjust NaN values
        logger.info("Handling NaN values by filling with appropriate values")
        merged_df["AWARD"] = merged_df["AWARD"].fillna("No")
        merged_df["BIPOC PRE-QUALIFIED"] = merged_df["BIPOC PRE-QUALIFIED"].fillna("No")
        merged_df["NEW CONSTRUCTION SET ASIDE"] = merged_df["NEW CONSTRUCTION SET ASIDE"].fillna(
            "none"
        )
        merged_df["NC POOL SELECTION: HOMELESS, ELI/VLI, MIP"] = merged_df[
            "NC POOL SELECTION: HOMELESS, ELI/VLI, MIP"
        ].fillna("none")
        merged_df["GP1 PARENT ORGANIZATION"] = merged_df["GP1 PARENT ORGANIZATION"].fillna("none")
        merged_df["GP2 COMPANY"] = merged_df["GP2 COMPANY"].fillna("none")
        merged_df["GP2 CONTACT"] = merged_df["GP2 CONTACT"].fillna("none")
        merged_df["GP2 PARENT COMPANY"] = merged_df["GP2 PARENT COMPANY"].fillna("none")
        merged_df["GP3 COMPANY"] = merged_df["GP3 COMPANY"].fillna("none")
        merged_df["GP3 CONTACT"] = merged_df["GP3 CONTACT"].fillna("none")
        merged_df["GP3 PARENT COMPANY"] = merged_df["GP3 PARENT COMPANY"].fillna("none")
        logger.info(f"After NaN handling, dataset has {len(merged_df)} rows")

        # Adjust column data types
        logger.info("Adjusting column data types")
        col = "EXCEEDING MINIMUM INCOME RESTRICTIONS (20 PTS)"
        if (merged_df[col] % 1 != 0).any():
            print(f"Warning: {col} has non-integer values; rounding down")
            merged_df[col] = merged_df[col].round(0)
        merged_df[col] = merged_df[col].astype("int64")
        logger.info(f"Adjusted column '{col}' to integer type")

        # Save processed data
        logger.info(f"Saving merged dataset to {output_path}")
        merged_df.to_csv(output_path, index=False)
        logger.success(f"Processing complete. Saved to {output_path}")

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise
    # -----------------------------------------


if __name__ == "__main__":
    app()
