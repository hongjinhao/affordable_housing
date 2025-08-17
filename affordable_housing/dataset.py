from pathlib import Path
import re

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import typer

from affordable_housing.config import EXTERNAL_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


def rename_column_names(applicant_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns in the DataFrame based on regular expression patterns.
    Args:
        applicant_df (pd.DataFrame): Input DataFrame to process.
    Returns:
        pd.DataFrame: DataFrame with renamed columns.
    Raises:
        AssertionError: If any regular expression matches more than one column.
    """
    num_of_matches = [0] * 20
    df = applicant_df.copy()  # Create a copy to avoid modifying the original

    for i, col in enumerate(df.columns):
        if re.match("average", col, re.IGNORECASE):
            df = df.rename(columns={col: "avg_targeted_affordability"})
            num_of_matches[0] += 1
        elif re.match("CDLAC TOTAL", col, re.IGNORECASE):
            df = df.rename(columns={col: "total_points"})
            num_of_matches[1] += 1
        elif re.search("tie-brea", col, re.IGNORECASE):
            df = df.rename(columns={col: "tie_breaker_self_score"})
            num_of_matches[2] += 1
        elif re.search("bond", col, re.IGNORECASE):
            df = df.rename(columns={col: "bond_request_amount"})
            num_of_matches[3] += 1
        elif re.search("units for homeless", col, re.IGNORECASE):
            df = df.rename(columns={col: "num_homeless_units"})
            num_of_matches[4] += 1
        elif re.search("construction type", col, re.IGNORECASE):
            df = df.rename(columns={col: "construction_type"})
            num_of_matches[5] += 1
        elif re.search("housing type", col, re.IGNORECASE):
            df = df.rename(columns={col: "housing_type"})
            num_of_matches[6] += 1
        elif re.search("CDLAC.*region", col, re.IGNORECASE):
            df = df.rename(columns={col: "CDLAC_region"})
            num_of_matches[7] += 1
        elif re.search("CDLAC.*pool", col, re.IGNORECASE):
            df = df.rename(columns={col: "CDLAC_pool"})
            num_of_matches[8] += 1
        elif re.search("BIPOC", col, re.IGNORECASE):
            df = df.rename(columns={col: "bipoc_binary"})
            num_of_matches[9] += 1
        elif re.match("new construction set aside", col, re.IGNORECASE):
            df = df.rename(columns={col: "new_construction_set_aside"})
            num_of_matches[10] += 1
        elif re.search("secondary new construction", col, re.IGNORECASE):
            df = df.rename(columns={col: "secondary_new_construction_set_aside"})
            num_of_matches[11] += 1
        elif re.search("application", col, re.IGNORECASE):
            df = df.rename(columns={col: "application_number"})
            num_of_matches[12] += 1

    # Ensure each regular expression matches at most one column
    for i in range(20):
        assert num_of_matches[i] < 2, (
            f"Pattern {i} matched {num_of_matches[i]} columns, expected at most 1"
        )

    return df


def clean_and_merge_columns(applicant_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and merge specific columns in the DataFrame, including set-aside and CDLAC pool columns.
    Args:
        applicant_df (pd.DataFrame): Input DataFrame to process.
    Returns:
        pd.DataFrame: DataFrame with cleaned and merged columns.
    """
    df = applicant_df.copy()  # Create a copy to avoid modifying the original

    # Handle new_construction_set_aside and secondary_new_construction_set_aside
    if "secondary_new_construction_set_aside" in df.columns:
        df["new_construction_set_aside"] = df["new_construction_set_aside"].fillna("")
        df["secondary_new_construction_set_aside"] = df[
            "secondary_new_construction_set_aside"
        ].fillna("")
        df["new_construction_set_aside"] = df["new_construction_set_aside"].str.upper()
        df["secondary_new_construction_set_aside"] = df[
            "secondary_new_construction_set_aside"
        ].str.upper()
        # Concatenate, avoiding extra commas for empty secondary values
        df["combined_set_aside"] = df["new_construction_set_aside"] + df[
            "secondary_new_construction_set_aside"
        ].apply(lambda x: f", {x}" if x else "")
        df["combined_set_aside"] = df["combined_set_aside"].str.strip(", ")
        df["combined_set_aside"] = df["combined_set_aside"].replace("", None)
    else:
        df["combined_set_aside"] = df["new_construction_set_aside"].str.upper()

    # Clean CDLAC_pool
    df["CDLAC_pool"] = df["CDLAC_pool"].str.upper()
    df["CDLAC_pool"] = df["CDLAC_pool"].str.strip(", ")

    # Handle combined_CDLAC_pool based on bipoc_binary
    if "bipoc_binary" in df.columns:
        df["combined_CDLAC_pool"] = np.where(
            df["bipoc_binary"] == "Yes", "BIPOC", df["CDLAC_pool"]
        )
    else:
        df["combined_CDLAC_pool"] = df["CDLAC_pool"]

    return df


def standardize_application_number(num: str) -> str:
    # Match CA-24-555 or 24-596 formats and transform them into CA-YYYY-digits
    match = re.match(r"(\w+)-(\d+)-(\d+)|(\d+)-(\d+)", num)
    if not match:
        return f"Invalid format: {num}"

    if match.group(1):  # CA-24-555 format
        prefix, year, seq = match.group(1), match.group(2), match.group(3)
        if prefix != "CA":
            return f"Invalid prefix: {num}"
    else:  # 24-596 format
        prefix, year, seq = "CA", match.group(4), match.group(5)

    # Convert year to four digits (assume 20XX for XX < 100)
    try:
        year_int = int(year)
        if year_int < 100:
            year = f"20{year_int:02d}"  # e.g., 24 -> 2024
        else:
            year = f"{year_int}"
    except ValueError:
        return f"Invalid year: {num}"

    return f"{prefix}-{year}-{seq}"


def clean_region(region: str) -> str:
    region = region.strip().lower()  # Normalize to lowercase for consistent matching

    if re.search(r"bay\s*area", region):
        return "BAY AREA"
    elif re.search(r"northern", region):
        return "NORTHERN"
    elif re.search(r"inland", region):
        return "INLAND"
    elif re.search(r"city\s*of\s*(la|los\s*angeles)", region):
        return "CITY OF LA"
    elif re.search(r"balance\s*of\s*(la|los\s*angeles)\s*county", region):
        return "BALANCE OF LA COUNTY"
    elif re.search(r"coastal", region):
        return "COASTAL"
    else:
        return "NONE"


def clean_construction_type(construction_type: str) -> str:
    construction_type = (
        construction_type.strip().lower()
    )  # Normalize to lowercase for consistent matching

    if re.search(r"acq", construction_type, re.IGNORECASE):
        return "ACQ AND REHAB"
    else:
        return construction_type.upper()


@app.command()
def main(
    # Input paths for applicant lists
    input_path_r1_2023_applicant: Path = EXTERNAL_DATA_DIR / "2023-R1-ApplicantList.xlsx",
    input_path_r2_2023_applicant: Path = EXTERNAL_DATA_DIR / "2023-R2-ApplicantList.xlsx",
    input_path_r3_2023_applicant: Path = EXTERNAL_DATA_DIR / "2023-R3-ApplicantList.xlsx",
    input_path_r1_2024_applicant: Path = EXTERNAL_DATA_DIR / "2024-R1-ApplicantList.xlsx",
    input_path_r2_2024_applicant: Path = EXTERNAL_DATA_DIR / "2024-R2-ApplicantList.xlsx",
    input_path_r1_2025_applicant: Path = EXTERNAL_DATA_DIR / "2025-R1-ApplicantList.xlsx",
    # Input paths for award lists
    input_path_labels_2023: Path = EXTERNAL_DATA_DIR / "2023-Financing-data.xlsx",
    input_path_labels_2024: Path = EXTERNAL_DATA_DIR / "2024-Financing-data.xlsx",
    input_path_labels_r1_2025: Path = EXTERNAL_DATA_DIR / "2025-R1-AwardList.xlsx",
    # Output
    output_path: Path = PROCESSED_DATA_DIR / "3yr_dataset.csv",
    output_path_train: Path = PROCESSED_DATA_DIR / "3yr_dataset_train.csv",
    output_path_test: Path = PROCESSED_DATA_DIR / "3yr_dataset_test.csv",
):
    """
    Combine datasets from 3 years (2023, 2024, 2025 till R1) by standardising their names, merging and cleaning.
    Split Dataset into train and test.
    """
    logger.info("Starting dataset processing...")

    try:
        applicant_data = [
            (input_path_r1_2023_applicant, "R1_2023_applicant"),
            (input_path_r2_2023_applicant, "R2_2023_applicant"),
            (input_path_r3_2023_applicant, "R3_2023_applicant"),
            (input_path_r1_2024_applicant, "R1_2024_applicant"),
            (input_path_r2_2024_applicant, "R2_2024_applicant"),
            (input_path_r1_2025_applicant, "R1_2025_applicant"),
        ]

        # List of award DataFrames with their paths and names
        award_data = [
            (input_path_labels_2023, "Labels_2023", 1),  # sheet_name=1 for 2023
            (input_path_labels_2024, "Labels_2024", 1),  # sheet_name=1 for 2024
            (input_path_labels_r1_2025, "Labels_R1_2025", 0),  # default sheet for 2025
        ]

        # Load applicant DataFrames
        applicant_dfs = []
        for path, name in applicant_data:
            logger.info(f"Loading applicant Excel file from {path}")
            df = pd.read_excel(path, header=1, index_col=None)
            df.attrs["file_name"] = str(path)  # Store file path in attrs
            logger.info(f"Loaded {name} with {len(df)} rows and {len(df.columns)} columns")
            applicant_dfs.append(df)

        # Load award DataFrames
        award_dfs = []
        for path, name, sheet in award_data:
            logger.info(f"Loading award Excel file from {path}")
            df = pd.read_excel(path, sheet_name=sheet)
            df.attrs["file_name"] = str(path)  # Store file path in attrs
            logger.info(f"Loaded {name} with {len(df)} rows and {len(df.columns)} columns")
            award_dfs.append(df)

        logger.info("Standardise excels, drop some empty rows, merging some columns and cleaning")
        applicant_df = pd.DataFrame()
        for df in applicant_dfs:
            df = rename_column_names(df)

            threshold = int(len(df.columns) * 0.1)
            df = df.dropna(thresh=threshold)
            df = clean_and_merge_columns(df)
            columns = [
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
            applicant_df = pd.concat([applicant_df, df[columns]])
        logger.info(
            f"Successfully processed applicant dataframes into one of size {applicant_df.shape}"
        )

        applicant_df["application_number"] = applicant_df["application_number"].apply(
            standardize_application_number
        )
        # Verify that all standardized numbers are 11 characters
        invalid_lengths = applicant_df["application_number"][
            applicant_df["application_number"].str.len() != 11
        ]
        if not invalid_lengths.empty:
            logger.warning(
                f"Warning: Some standardized numbers do not have 11 characters: {invalid_lengths}"
            )

        labels_df = pd.DataFrame()
        for df in award_dfs:
            for col in df.columns:
                if re.search(r"application|CTCAC", col, re.IGNORECASE):
                    df = df.rename(columns={col: "application_number"})
                    df["application_number"] = df["application_number"].apply(
                        standardize_application_number
                    )
            labels_df = pd.concat([labels_df, df["application_number"]])
        labels_df["award"] = "Yes"
        logger.info(f"Successfully create labels dataframe with size {labels_df.shape}")

        # Combine features and labels
        dataset = pd.merge(applicant_df, labels_df, how="left", on="application_number")
        dataset["award"] = dataset["award"].fillna("No")

        dataset["combined_set_aside"] = dataset["combined_set_aside"].fillna("NONE")
        dataset["CDLAC_region"] = dataset["CDLAC_region"].fillna("NONE")
        dataset["CDLAC_region"] = dataset["CDLAC_region"].apply(clean_region)
        dataset["construction_type"] = dataset["construction_type"].apply(clean_construction_type)
        logger.info(f"Successfully create dataset with size {dataset.shape}")

        # Save processed data

        train_df, test_df = train_test_split(
            dataset, test_size=0.20, stratify=dataset["award"], random_state=42
        )
        logger.info(
            f"Successfully split train and test. Train shape: {train_df.shape} and Test shape: {test_df.shape}"
        )

        dataset.to_csv(output_path, index=False)
        train_df.to_csv(output_path_train, index=False)
        test_df.to_csv(output_path_test, index=False)
        logger.success(
            f"Processing complete. Saved to {output_path}, {output_path_train} and {output_path_test}"
        )

    except Exception as e:
        logger.error(f"Error during processing: {str(e)}")
        raise
    # -----------------------------------------


if __name__ == "__main__":
    app()
