import logging
import os
import sys
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
import requests
from io import StringIO

# Add the utils directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from utils import get_data_location, download_data  # Ensure these functions are correctly implemented

class GenerateTrainingData:
    """
    Class to download and generate training data for the GNN Spatio-Temporal project.
    It downloads COVID-19 data, processes it, and saves it in raw and processed formats.
    """

    def __init__(self):
        self.df = None
        # Base URLs for raw data
        self.url_base_us = (
            "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
            "csse_covid_19_data/csse_covid_19_daily_reports_us"
        )
        self.url_base = (
            "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
            "csse_covid_19_data/csse_covid_19_daily_reports"
        )
        # Common columns expected in the dataset, including the new columns
        self.common_columns = [
            "state",
            "latitude",
            "longitude",
            "fips",
            "date_today",
            "confirmed",
            "deaths",
            "recovered",
            "active",
            "hospitalization",
            "new_cases",
            "hospitalization_rate",  # Added
            "mortality_rate",        # Added
            "case_fatality_ratio",   # Added
        ]

    def download_single_file(self, date):
        """
        Downloads a single CSV file from the JHU COVID-19 dataset for a specific date.
        Returns a cleaned and processed DataFrame, and saves raw data.

        :param date: Date string in 'YYYY-MM-DD' format.
        :return: Processed DataFrame or None if download fails.
        """
        date_formats = ["%m-%d-%Y", "%m-%d-%y"]
        try:
            # Generate formatted dates
            formatted_dates = [datetime.strptime(date, "%Y-%m-%d").strftime(fmt) for fmt in date_formats]
        except ValueError as ve:
            logging.error(f"Date formatting error for {date}: {ve}")
            return None

        logging.info(f"Processing date: {date} with formatted dates: {formatted_dates}")

        # Attempt to download from both 'daily_reports_us' and 'daily_reports'
        for base_url in [self.url_base_us, self.url_base]:
            for fmt, formatted_date in zip(date_formats, formatted_dates):
                url = f"{base_url}/{formatted_date}.csv"
                logging.info(f"Attempting to download data from URL: {url}")
                data = download_data(url=url)

                if data is not None:
                    try:
                        # Save raw data to the 'raw' directory
                        folder = "raw"
                        raw_file_path = get_data_location(f"{formatted_date}_raw.csv", folder=folder)
                        data.to_csv(raw_file_path, index=False)
                        logging.info(f"Raw data saved to {raw_file_path}")

                        # Add 'date_today' column based on the date format
                        data["date_today"] = datetime.strptime(formatted_date, fmt)

                        # Rename columns with conditional handling
                        rename_dict = {
                            "Province_State": "state",
                            "Lat": "latitude",
                            "Long_": "longitude",
                            "Confirmed": "confirmed",
                            "Deaths": "deaths",
                            "Recovered": "recovered",
                            "Active": "active",
                            "FIPS": "fips",
                            "Date": "date_today",  # Ensure 'Date' is mapped if present
                            "Hospitalization_Rate": "hospitalization_rate",     # Added
                            "Mortality_Rate": "mortality_rate",                 # Added
                            "Case_Fatality_Ratio": "case_fatality_ratio",       # Added
                        }

                        # Check if 'People_Hospitalized' exists before renaming
                        if "People_Hospitalized" in data.columns:
                            rename_dict["People_Hospitalized"] = "hospitalization"
                        else:
                            # Use NaN to allow for better handling later
                            data["hospitalization"] = np.nan
                            logging.warning(f"'People_Hospitalized' missing in {formatted_date}.csv")

                        data = data.rename(columns=rename_dict).dropna(subset=["fips"])

                        # Convert 'fips' to integer
                        data["fips"] = data["fips"].astype(int)

                        # Ensure all common columns are present
                        for col in self.common_columns:
                            if col not in data.columns:
                                # Assign NaN for missing columns to allow for proper handling
                                data[col] = np.nan
                                logging.warning(f"Column '{col}' missing in {formatted_date}.csv. Filled with NaN.")

                        # Select only the common columns
                        data = data[self.common_columns]

                        return data

                    except Exception as e:
                        logging.error(f"Error processing {formatted_date}.csv: {e}")
                        return None
                else:
                    logging.info(f"{formatted_date}.csv doesn't exist or failed to download from {base_url}!")

        # If all attempts fail
        logging.warning(f"Data for {date} is missing in all attempted formats and folders.")
        return None

    def download_jhu_data(self, start_time, end_time):
        """
        Downloads and processes COVID-19 data for a given date range.
        Saves raw and processed data.
        Returns a combined DataFrame of all the processed data.

        :param start_time: Start date in 'YYYY-MM-DD' format.
        :param end_time: End date in 'YYYY-MM-DD' format.
        :return: Combined processed DataFrame or None if download fails.
        """
        try:
            # Generate list of dates in 'YYYY-MM-DD' format
            date_list = pd.date_range(start_time, end_time).strftime("%Y-%m-%d")
            logging.info(f"Generated date list from {start_time} to {end_time}")

            # Use multiprocessing to download data files in parallel
            with Pool(processes=10) as pool:  # Limit to 10 parallel processes
                data = pool.map(self.download_single_file, date_list)
            logging.info("Finished downloading data.")

            # Log missing dates
            missing_dates = [date for date, d in zip(date_list, data) if d is None]
            if missing_dates:
                logging.warning(f"Missing data for dates: {missing_dates}")

            # Filter out None entries and concatenate the data
            data = [x for x in data if x is not None]
            if not data:
                logging.error(
                    "No data was downloaded. Please check the date range and data source."
                )
                return None
            data = pd.concat(data, axis=0)
            logging.info(f"Concatenated data shape: {data.shape}")

            # Convert 'date_today' to datetime if not already
            data["date_today"] = pd.to_datetime(data["date_today"])

            # Sort data by fips and date to ensure correct diff operations
            data = data.sort_values(by=["fips", "date_today"])

            # Calculate daily new cases
            data["new_cases"] = data.groupby("fips")["confirmed"].diff().fillna(0)
            data["new_cases"] = data["new_cases"].apply(lambda x: x if x >= 0 else 0)

            # Calculate daily hospitalizations
            # Handle missing hospitalization data by forward filling within each fips group
            data["hospitalization"] = data.groupby("fips")["hospitalization"].ffill().fillna(0)
            data["hospitalization"] = data.groupby("fips")["hospitalization"].diff().fillna(0)
            data["hospitalization"] = data["hospitalization"].apply(
                lambda x: x if x >= 0 else 0
            )

            # Ensure all common columns are present
            for col in self.common_columns:
                if col not in data.columns:
                    data[col] = 0  # Assign default value for missing columns
                    logging.warning(f"Column '{col}' missing in the concatenated data. Filled with 0.")

            # Select only the common columns
            data = data[self.common_columns]

            # Fill remaining NaNs if any
            data = data.fillna(0)

            # Save the processed data to a pickle file in the 'processed' directory
            processed_file_path = get_data_location(
                "processed_covid_data.pickle", folder="processed"
            )
            data.to_pickle(processed_file_path)
            logging.info(f"Processed data saved to {processed_file_path}")

            self.df = data
            return data

        except Exception as e:
            logging.error(f"An error occurred during data download and processing: {e}")
            return None

if __name__ == "__main__":
    # Configure logging with detailed format and file handler
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("data_generation.log")
        ]
    )
    # Initialize the data generator
    generator = GenerateTrainingData()
    # Define the date range for data collection
    start_date = "2020-04-01"
    end_date = "2023-12-31"
    # Download and process the training data
    generator.download_jhu_data(start_date, end_date)
