import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Import get_data_location from src.utils.utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from utils import get_data_location

# Load the processed data
processed_file_path = get_data_location("processed_covid_data.pickle", folder="processed")
processed_data = pd.read_pickle(processed_file_path)

# Display the first few rows
print(processed_data.head())

# Ensure 'date_today' is in datetime format
processed_data['date_today'] = pd.to_datetime(processed_data['date_today'])

# Plot the data for Alaska for confirmed cases
def plot_alaska_confirmed_cases(data):
    alaska_data = data[data["state"] == "Alaska"]
    plt.figure(figsize=(12, 6))
    plt.plot(alaska_data["date_today"], alaska_data["confirmed"], marker="o")
    plt.title("Confirmed COVID-19 cases in Alaska")
    plt.xlabel("Date")
    plt.ylabel("Confirmed cases")
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_alaska_confirmed_cases(processed_data)

# Plot confirmed cases over time for all states
def plot_confirmed_cases_over_time(data):
    plt.figure(figsize=(14, 7))
    sns.lineplot(data=data, x='date_today', y='confirmed', hue='state', legend=False)
    plt.title('Confirmed COVID-19 Cases Over Time by State')
    plt.xlabel('Date')
    plt.ylabel('Confirmed Cases')
    plt.show()

plot_confirmed_cases_over_time(processed_data)

# Display data information and statistics
print(processed_data.info())
print(processed_data.describe())
print(processed_data.isnull().sum())

# Plotting confirmed cases for a specific date
def plot_confirmed_cases_specific_date(data, date):
    data_specific_date = data[data['date_today'] == date]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data_specific_date, x='longitude', y='latitude', size='confirmed', hue='state', alpha=0.6)
    plt.title(f'Confirmed COVID-19 Cases on {date}')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

plot_confirmed_cases_specific_date(processed_data, '2020-04-12')

# Compute and plot the correlation matrix
def plot_correlation_matrix(data):
    corr_matrix = data[['confirmed', 'deaths', 'recovered', 'active', 'hospitalization', 'new_cases']].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of COVID-19 Metrics')
    plt.show()

plot_correlation_matrix(processed_data)

# Plot confirmed cases over time for a specific state
def plot_state_confirmed_cases(data, state):
    state_data = data[data['state'] == state]
    plt.figure(figsize=(12, 6))
    plt.plot(state_data['date_today'], state_data['confirmed'], marker='o', label='Confirmed Cases')
    plt.title(f'Confirmed COVID-19 Cases Over Time in {state}')
    plt.xlabel('Date')
    plt.ylabel('Confirmed Cases')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

plot_state_confirmed_cases(processed_data, 'California')


################################################# Load all the pickled files in data folder #################################################

