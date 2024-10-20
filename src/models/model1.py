import pandas as pd
import os
import sys

# import get_data_location from src.utils.utils 
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from utils import get_data_location

# Load the processed data
processed_file_path = get_data_location("processed_covid_data.pickle", folder="processed")
processed_data = pd.read_pickle(processed_file_path)

# Display the first few rows
processed_data.head()

# plot the data for alasaka for confirmed cases
import matplotlib.pyplot as plt

# Filter data for Alaska
alaska_data = processed_data[processed_data["state"] == "Alaska"]

# Plot the confirmed cases
plt.figure(figsize=(12, 6))
plt.plot(alaska_data["date_today"], alaska_data["confirmed"], marker="o")
plt.title("Confirmed COVID-19 cases in Alaska")
plt.xlabel("Date")
plt.ylabel("Confirmed cases")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure 'date_today' is in datetime format
processed_data['date_today'] = pd.to_datetime(processed_data['date_today'])

# Plot confirmed cases over time for all states
plt.figure(figsize=(14, 7))
sns.lineplot(data=processed_data, x='date_today', y='confirmed', hue='state', legend=False)
plt.title('Confirmed COVID-19 Cases Over Time by State')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.show()


processed_data.info()
processed_data.describe()
processed_data.isnull().sum()

# Plotting confirmed cases for a specific date
specific_date = '2020-04-12'
data_specific_date = processed_data[processed_data['date_today'] == specific_date]

plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_specific_date, x='longitude', y='latitude', size='confirmed', hue='state', alpha=0.6)
plt.title(f'Confirmed COVID-19 Cases on {specific_date}')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# Compute correlation matrix
corr_matrix = processed_data[['confirmed', 'deaths', 'recovered', 'active', 'hospitalization', 'new_cases']].corr()

# Plot the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of COVID-19 Metrics')
plt.show()


# Select a state, e.g., California
state = 'California'
state_data = processed_data[processed_data['state'] == state]

# Plot confirmed cases over time
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
