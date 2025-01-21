# EpiGNN/data_merge.py

#!/usr/bin/env python3
"""
data_merge.py
-------------
Utilities for merging, cleaning, and correcting COVID-19 daily data.
"""

import os
import logging
import pandas as pd

# Example reference coordinates for your project
REFERENCE_COORDINATES = {
    "East of England": (52.1766, 0.425889),
    "Midlands": (52.7269, -1.458210),
    "London": (51.4923, -0.308660),
    "South East": (51.4341, -0.969570),
    "South West": (50.8112, -3.633430),
    "North West": (53.8981, -2.657550),
    "North East and Yorkshire": (54.5378, -2.180390),
}


def load_and_correct_data_daily(data: pd.DataFrame, ref_coords: dict = REFERENCE_COORDINATES) -> pd.DataFrame:
    """
    Assign latitude and longitude to each region, sort data, and fill missing values.

    Parameters:
    - data (pd.DataFrame): Raw COVID-19 data.
    - ref_coords (dict): Reference coordinates for regions.

    Returns:
    - pd.DataFrame: Processed data.
    """
    # Assign latitude and longitude based on reference coordinates
    for region, coords in ref_coords.items():
        data.loc[data["areaName"] == region, ["latitude", "longitude"]] = coords

    # Fill NAs for daily features with 0
    daily_feats = ["new_confirmed", "new_deceased", "newAdmissions", "hospitalCases", "covidOccupiedMVBeds"]
    data[daily_feats] = data[daily_feats].fillna(0)

    # Sort data by region and date
    data.sort_values(["areaName", "date"], inplace=True)

    logging.info("Data loaded and coordinates assigned. Sorted by areaName and date.")
    return data


def merge_and_save(raw_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    """
    Load raw data, process it, and save the cleaned data.

    Parameters:
    - raw_csv_path (str): Path to the raw merged CSV data.
    - output_csv_path (str): Path to save the processed daily data.

    Returns:
    - pd.DataFrame: Processed data.
    """
    if not os.path.exists(raw_csv_path):
        logging.error(f"Raw CSV file not found: {raw_csv_path}")
        raise FileNotFoundError(f"Raw CSV file not found: {raw_csv_path}")

    # Load raw data
    df_raw = pd.read_csv(raw_csv_path, parse_dates=["date"])

    # Process data
    df_processed = load_and_correct_data_daily(df_raw)

    # Save processed data
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_processed.to_csv(output_csv_path, index=False)
    logging.info(f"Processed daily data saved -> {output_csv_path}")

    return df_processed


if __name__ == "__main__":
    # Example usage
    import argparse

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Merge and preprocess COVID-19 daily data.")
    parser.add_argument("--raw_csv", type=str, required=True, help="Path to raw merged CSV data.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save processed daily data.")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Merge and save data
    merge_and_save(args.raw_csv, args.output_csv)
