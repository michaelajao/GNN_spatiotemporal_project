# src/utils/utils.py

import io
import logging
import os
import pytz
from datetime import datetime

import numpy as np
import pandas as pd
import requests
from epiweeks import Week
from haversine import haversine

# Disable warnings from urllib3 (use with caution)
requests.packages.urllib3.disable_warnings()

# Configure logging
logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

DATE_FORMAT = "%Y-%m-%d"
splitter = "|"
date_today = datetime.now(tz=pytz.timezone("US/Eastern")).strftime(DATE_FORMAT)

# Define the base data directory relative to the project root
# Assuming 'utils.py' is located at 'src/utils/utils.py'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def get_data_location(file_name, folder=None):
    """
    Constructs the file path to store or read data.

    :param file_name: The name of the file (e.g., 'state_covid_data.pickle').
    :param folder: Subfolder to save or read the file from ('raw' or 'processed').
                   It ensures data is saved in 'data/raw' or 'data/processed' directories.
    :return: Full path to the file.
    """
    if folder == "raw":
        folder_path = os.path.join(DATA_DIR, "raw")
    elif folder == "processed":
        folder_path = os.path.join(DATA_DIR, "processed")
    else:
        raise ValueError("Invalid folder name. Use 'raw' or 'processed'.")

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logging.info(f"Created directory: {folder_path}")

    # Construct the full file path
    file_path = os.path.join(folder_path, file_name)
    logging.info(f"File path set to: {file_path}")
    return file_path


def check_url(url):
    """
    Checks the existence of a URL.

    :param url: The URL to check.
    :return: True if the URL exists, False otherwise.
    """
    try:
        request = requests.get(url, verify=False)
        if request.status_code < 400:
            return True
        else:
            logging.info(f"URL for {url.split('/')[-1]} does not exist!")
            return False
    except requests.exceptions.RequestException as e:
        logging.error(f"Request exception occurred: {e}")
        return False


def download_data(url):
    """
    Downloads CSV data from a URL and returns it as a Pandas DataFrame.

    :param url: The URL to download the CSV data from.
    :return: Pandas DataFrame if download is successful, None otherwise.
    """
    if check_url(url):
        try:
            response = requests.get(url=url, verify=False)
            response.raise_for_status()
            df = pd.read_csv(io.StringIO(response.content.decode("utf-8")))
            logging.info(f"Data downloaded successfully from {url}")
            return df
        except Exception as e:
            logging.error(f"Failed to download data from {url}: {e}")
    return None


def calculate_ccc(y_true, y_pred):
    """
    Calculates the concordance correlation coefficient (CCC) between two vectors.

    :param y_true: Real data.
    :param y_pred: Estimated data.
    :return: CCC value.
    """
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    numerator = 2 * cor * np.sqrt(var_true) * np.sqrt(var_pred)
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator


def gravity_law_commute_dist(lat1, lng1, pop1, lat2, lng2, pop2, r=1e4):
    """
    Calculates the edge weight based on the gravity law for commute distance.

    :param lat1: Latitude of location 1.
    :param lng1: Longitude of location 1.
    :param pop1: Population of location 1.
    :param lat2: Latitude of location 2.
    :param lng2: Longitude of location 2.
    :param pop2: Population of location 2.
    :param r: Radius parameter for decay, default is 10,000 km.
    :return: Edge weight.
    """
    d = haversine((lat1, lng1), (lat2, lng2), unit="km")
    alpha = 0.1
    beta = 0.1
    # Prevent division by zero by adding a small epsilon
    weight = (np.exp(-d / r)) / (np.abs((pop1**alpha) - (pop2**beta)) + 1e-5)
    return weight


def envelope(x):
    """
    Calculates the envelope of a signal.

    :param x: Input data as a NumPy array or list.
    :return: Envelope of the signal.
    """
    x = np.array(x).copy()
    for i in range(len(x) - 1):
        if x[i + 1] < x[i]:
            x[i + 1] = x[i]
    return x


def map_to_week(df, date_column="date_today", groupby_target=None):
    """
    Maps 'date_today' to the corresponding week ending date.

    :param df: Pandas DataFrame.
    :param date_column: Column name containing dates.
    :param groupby_target: Column(s) to group by and sum over.
    :return: DataFrame with week_id.
    """
    df[date_column] = df[date_column].apply(
        lambda x: Week.fromdate(x).enddate() if pd.notna(x) else x
    )
    df[date_column] = pd.to_datetime(df[date_column])
    if groupby_target is not None:
        df = df.groupby(date_column, as_index=False)[groupby_target].sum()
    return df


def normalize_features(df, feature_columns):
    """
    Normalizes specified feature columns in the DataFrame using Min-Max scaling.

    :param df: Pandas DataFrame.
    :param feature_columns: List of column names to normalize.
    :return: DataFrame with normalized features.
    """
    for col in feature_columns:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val != 0:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0
    return df
