import logging
import os
import sys
from datetime import datetime
from multiprocessing import Pool
from urllib.parse import urljoin
import io  # Required for StringIO

import numpy as np
import pandas as pd
import requests

# Add the utils directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from utils.utils import get_data_location  # Ensure this function is correctly implemented

def download_data(url):
    """
    Downloads CSV data from a given URL and returns a pandas DataFrame.
    
    :param url: URL of the CSV file.
    :return: pandas DataFrame or None if download fails.
    """
    try:
        response = requests.get(url, timeout=10)  # Added timeout
        if response.status_code == 200:
            # Use io.StringIO instead of pd.compat.StringIO
            data = pd.read_csv(io.StringIO(response.text))
            logging.info(f"Data downloaded successfully from {url}")
            return data
        else:
            logging.warning(f"URL for {url} returned status code {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Exception occurred while downloading data from {url}: {e}")
        return None

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
            "csse_covid_19_data/csse_covid_19_daily_reports_us/"
        )
        self.url_base = (
            "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/"
            "csse_covid_19_data/csse_covid_19_daily_reports/"
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
        formatted_dates = []
        for fmt in date_formats:
            try:
                formatted_date = datetime.strptime(date, "%Y-%m-%d").strftime(fmt)
                formatted_dates.append(formatted_date)
            except ValueError:
                logging.error(f"Date formatting error for {date} with format {fmt}")
                continue

        if not formatted_dates:
            logging.error(f"No valid date formats for date: {date}")
            return None

        logging.info(f"Processing date: {date} with formatted dates: {formatted_dates}")

        # Attempt to download from both 'daily_reports_us' and 'daily_reports'
        for base_url in [self.url_base_us, self.url_base]:
            for formatted_date in formatted_dates:
                url = urljoin(base_url, f"{formatted_date}.csv")
                logging.info(f"Attempting to download data from URL: {url}")
                data = download_data(url=url)

                if data is not None:
                    try:
                        # Save raw data to the 'raw' directory
                        folder = "raw"
                        raw_file_path = get_data_location(f"{formatted_date}_raw.csv", folder=folder)
                        data.to_csv(raw_file_path, index=False)
                        logging.info(f"Raw data saved to {raw_file_path}")

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

                        # Rename columns and drop rows where 'fips' is missing
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

                        # Add 'date_today' column using the original date
                        data["date_today"] = pd.to_datetime(date)

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
        Saves processed data.
        Returns a combined DataFrame of all the processed data.

        :param start_time: Start date in 'YYYY-MM-DD' format.
        :param end_time: End date in 'YYYY-MM-DD' format.
        :return: Combined processed DataFrame or None if download fails.
        """
        try:
            # Generate list of dates in 'YYYY-MM-DD' format
            date_list = pd.date_range(start_time, end_time).strftime("%Y-%m-%d").tolist()
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
            logging.info("Data sorted by 'fips' and 'date_today'.")

            # Calculate daily new cases
            data["new_cases"] = data.groupby("fips")["confirmed"].diff().fillna(0)
            data["new_cases"] = data["new_cases"].apply(lambda x: x if x >= 0 else 0)
            logging.info("Calculated 'new_cases'.")

            # Calculate daily hospitalizations
            # Handle missing hospitalization data by forward filling within each fips group
            data["hospitalization"] = data.groupby("fips")["hospitalization"].ffill().fillna(0)
            data["hospitalization"] = data.groupby("fips")["hospitalization"].diff().fillna(0)
            data["hospitalization"] = data["hospitalization"].apply(
                lambda x: x if x >= 0 else 0
            )
            logging.info("Calculated 'hospitalization'.")

            # Ensure all common columns are present
            for col in self.common_columns:
                if col not in data.columns:
                    data[col] = 0  # Assign default value for missing columns
                    logging.warning(f"Column '{col}' missing in the concatenated data. Filled with 0.")

            # Select only the common columns
            data = data[self.common_columns]
            logging.info("Selected common columns.")

            # Fill remaining NaNs if any
            data = data.fillna(0)
            logging.info("Filled remaining NaNs with 0.")

            # Save the processed data to a CSV file in the 'processed' directory
            processed_file_path = get_data_location(
                "processed_covid_data.csv", folder="processed"
            )
            data.to_csv(processed_file_path, index=False)
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
