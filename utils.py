# FILE: utils.py

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

requests.packages.urllib3.disable_warnings()

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

DATE_FORMAT = "%Y-%m-%d"
splitter = "|"
date_today = datetime.now(tz=pytz.timezone('US/Eastern')).strftime(DATE_FORMAT)
DATA_LOCATION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def get_data_location(file_name, folder=None):
    """
        This is the function that generates the path for file to be read
        :param folder: folder name
        :type folder: str
        :param file_name: file name
        :type file_name: str
        :return: path of the file
        :rtype: str
        """
    return os.path.join(DATA_LOCATION, file_name) if folder is None else os.path.join(DATA_LOCATION, folder, file_name)

def check_url(url):
    """
    Function to check the existence of url
    :param url:
    :return:
    """
    try:
        request = requests.get(url, verify=False)
        if request.status_code < 400:
            return True
        else:
            logging.info(f"URL for {url.split('/')[-1]} does not exist!")
            return False
    except Exception as e:
        logging.info(f"Error checking URL {url}: {e}")
        return False

def download_data(url):
    """
    Function that downloads the csv files from Github
    :param url: url of the csv file
    :type url: str
    :return: content of csv file
    :rtype: pandas.DataFrame
    """
    if check_url(url):
        try:
            x = requests.get(url=url, verify=False).content
            df = pd.read_csv(io.StringIO(x.decode('utf8')))
            return df
        except Exception as e:
            logging.info(f"Error downloading data from {url}: {e}")
            return None

def calculate_ccc(y_true, y_pred):
    """
    This function calculates the concordance correlation coefficient (CCC) between two vectors
    :param y_true: real data
    :param y_pred: estimated data
    :return: CCC
    :rtype: float
    """
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator if denominator != 0 else 0

def gravity_law_commute_dist(lat1, lng1, pop1, lat2, lng2, pop2, r=1):
    """
    This function calculates the edge via the gravity law
    :param lat1: latitude of location 1
    :type lat1: float
    :param lng1: longitude of location 1
    :type lng1: float
    :param pop1: population of location 1
    :type pop1: float or int
    :param lat2: latitude of location 2
    :type lat2: float
    :param lng2: longitude of location 2
    :type lng2: float
    :param pop2: population of location 2
    :type pop2: float or int
    :param r: decay rate, by default 1
    :type r: float or int
    :return: edge value
    :rtype: float or int
    """
    d = haversine((lat1, lng1), (lat2, lng2), unit='km')
    alpha = 0.1
    beta = 0.1
    r = 1e4

    w = (np.exp(-d / r)) / (abs((pop1 ** alpha) - (pop2 ** beta)) + 1e-5)
    return w

def envelope(x):
    """
    Function to calculate the envelope of a signal
    :param x: input data
    :type x: numpy.array or list
    :return: envelope of a signal
    :rtype: numpy.array or list
    """
    x = x.copy()
    for i in range(len(x) - 1):
        a = x[i]
        b = x[i + 1]
        if b < a:
            x[i + 1] = a
    return x

def map_to_week(df, date_column='date_today', groupby_target=None):
    """
    Map date_today to week_id

    Args:
        df (pd.DataFrame): Input dataframe.
        date_column (str, optional): Column name related to date_today. Defaults to 'date_today'.
        groupby_target (None or str or list, optional): Group by date_today and sum over the groupby_target. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with week_id.
    """
    df[date_column] = df[date_column].apply(lambda x: Week.fromdate(x).enddate() if pd.notna(x) else x)
    df[date_column] = pd.to_datetime(df[date_column])
    if groupby_target is not None:
        df = df.groupby(date_column, as_index=False)[groupby_target].sum()
    return df
