import os
import sys
import pandas as pd
import numpy as np
import pickle
from tqdm.auto import tqdm

# Set up directories and helper imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from utils import get_data_location, gravity_law_commute_dist, map_to_week

# Constants
START_DATE = '2020-06-01'
END_DATE = '2022-12-01'
FEATURE_COLUMNS = ['latitude', 'longitude', 'fips', 'confirmed', 'deaths', 'recovered', 'active', 'hospitalization', 'new_cases']

# Load Data
covid_data_path = '../../data/processed/processed_covid_data.pickle'
with open(covid_data_path, 'rb') as f:
    raw_data = pickle.load(f)

# Load and preprocess population data
pop_data_path = '../../data/uszips.csv'
pop_data = pd.read_csv(pop_data_path)
pop_data = pop_data.groupby('state_name').agg({
    'population': 'sum',
    'density': 'mean'
}).reset_index()

# Merge COVID data with population data
raw_data = pd.merge(
    raw_data,
    pop_data,
    how='inner',
    left_on='state',
    right_on='state_name'
)

# Data Preprocessing
# Convert 'date_today' to datetime
raw_data['date_today'] = pd.to_datetime(raw_data['date_today'])

# Fill missing values in 'hospitalization' column
raw_data['hospitalization'] = raw_data['hospitalization'].ffill().fillna(0)

# Filter data within the specified date range
raw_data = raw_data[
    (raw_data['date_today'] >= pd.to_datetime(START_DATE)) & 
    (raw_data['date_today'] <= pd.to_datetime(END_DATE))
].reset_index(drop=True)

# Create daily and weekly data frames
daily_data = raw_data.copy().drop('state_name', axis=1) # Drop redundant column
weekly_data = map_to_week(raw_data, date_column='date_today', groupby_target=FEATURE_COLUMNS)

# Re-merge population data for weekly data (if necessary)
state_population = pop_data[['state_name', 'population', 'density']].rename(columns={'state_name': 'state'})
weekly_data = pd.merge(weekly_data, state_population, on='state', how='left')

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def plot_weekly_hospitalizations(data, state='Alaska'):
    """
    Plots the weekly hospitalization cases for a specified state with research-quality aesthetics.

    :param data: Pandas DataFrame containing the data.
    :param state: The state to plot data for.
    """
    state_data = data[data['state'] == state]
    
    # Plot setup
    plt.figure(figsize=(12, 7))
    plt.plot(
        state_data['date_today'], 
        state_data['hospitalization'], 
        linestyle='-', 
        linewidth=1.5,
        markersize=4,
        color='#1f77b4'
    )
    
    # Title and labels
    plt.title(f'Weekly Hospitalization Cases for {state}', fontsize=16, weight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Hospitalization Cases', fontsize=14)
    
    # Date formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    # Grid and legend
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# Example usage
plot_weekly_hospitalizations(weekly_data, state='Alaska')

def plot_daily_cases(data, state='Alaska'):
    """
    Plots the daily cases for a specified state with research-quality aesthetics.

    :param data: Pandas DataFrame containing the data.
    :param state: The state to plot data for.
    """
    state_data = data[data['state'] == state]
    
    # Plot setup
    plt.figure(figsize=(12, 7))
    plt.plot(
        state_data['date_today'], 
        state_data['hospitalization'], 
        linestyle='-', 
        linewidth=1.5,
        markersize=4,
        color='#ff7f0e'
    )
    
    # Title and labels
    plt.title(f'Daily New Cases for {state}', fontsize=16, weight='bold')
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('New Cases', fontsize=14)
    
    # Date formatting
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha='right')
    
    # Grid and legend
    plt.grid(visible=True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

# Example usage
plot_daily_cases(daily_data, state='Alaska')


states = daily_data['state'].unique()
state_indices = {state: idx for idx, state in enumerate(states)}
num_nodes = len(states)

# Extract static features and remove duplicates
static_features = daily_data[['state', 'population', 'density', 'lat', 'lng']].drop_duplicates('state')
static_features.set_index('state', inplace=True)

from itertools import combinations

# Create dictionaries for quick access
state_latitudes = static_features['lat'].to_dict()
state_longitudes = static_features['lng'].to_dict()
state_populations = static_features['population'].to_dict()

# Initialize edge lists
edge_list = []
edge_weights = []

# Generate all possible pairs of states
state_pairs = list(combinations(states, 2))

for (state1, state2) in state_pairs:
    idx1 = state_indices[state1]
    idx2 = state_indices[state2]
    
    lat1, lng1 = state_latitudes[state1], state_longitudes[state1]
    lat2, lng2 = state_latitudes[state2], state_longitudes[state2]
    pop1, pop2 = state_populations[state1], state_populations[state2]
    
    weight = gravity_law_commute_dist(lat1, lng1, pop1, lat2, lng2, pop2)
    
    edge_list.append([idx1, idx2])
    edge_weights.append(weight)
    # Since the graph is undirected, add both directions
    edge_list.append([idx2, idx1])
    edge_weights.append(weight)

# Convert to tensors
edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weights, dtype=torch.float)

# Convert date column to datetime
daily_data['date_today'] = pd.to_datetime(daily_data['date_today'])

# Get sorted list of dates
dates = np.sort(daily_data['date_today'].unique())

# Number of features
num_static_features = 4  # latitude, longitude, population size, density
num_dynamic_features = 4  # active, confirmed, hospitalization, new_cases
num_features = num_static_features + num_dynamic_features

# Initialize list to store graph snapshots
graph_snapshots = []

for date in dates:
    # Filter data for the current date
    date_data = daily_data[daily_data['date_today'] == date]
    date_data.set_index('state', inplace=True)
    
    # Initialize feature matrix and label vector
    x = torch.zeros((num_nodes, num_features), dtype=torch.float)
    y = torch.zeros(num_nodes, dtype=torch.float)  # Labels (hospitalization)
    
    for state in states:
        idx = state_indices[state]
        
        # Static features
        static_feat = static_features.loc[state].values  # [population, density, lat, lng]
        
        # Dynamic features
        if state in date_data.index:
            dynamic_feat = date_data.loc[state][['active', 'confirmed', 'hospitalization', 'new_cases']].values.astype(float)
            # Handle missing values
            dynamic_feat = np.nan_to_num(dynamic_feat)
            # Label: Hospitalizations (proxy for ICU beds demand)
            y[idx] = dynamic_feat[2]  # hospitalization
        else:
            dynamic_feat = np.zeros(num_dynamic_features)
            y[idx] = 0.0  # No data for this state on this date
        
        # Combine static and dynamic features
        features = np.concatenate([static_feat, dynamic_feat])
        x[idx] = torch.tensor(features, dtype=torch.float)
    
    # Create Data object for PyTorch Geometric
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)
    data.date = date  # Store the date for reference
    
    # Append to the list of graph snapshots
    graph_snapshots.append(data)

history_window = 7  # Use past 7 days
pred_window = 1     # Predict next day's hospitalization

from torch.utils.data import Dataset, DataLoader

class TemporalGraphDataset(Dataset):
    def __init__(self, graph_snapshots, history_window, pred_window):
        self.graph_snapshots = graph_snapshots
        self.history_window = history_window
        self.pred_window = pred_window

    def __len__(self):
        return len(self.graph_snapshots) - (self.history_window + self.pred_window) + 1

    def __getitem__(self, idx):
        # Get input sequence
        x_seq = []
        y_seq = []
        for i in range(idx, idx + self.history_window):
            data = self.graph_snapshots[i]
            x_seq.append(data.x)
            y_seq.append(data.y)
        
        # Stack sequences
        x_seq = torch.stack(x_seq)  # Shape: [history_window, num_nodes, num_features]
        y_seq = torch.stack(y_seq)  # Shape: [history_window, num_nodes]
        
        # Target labels (next day's hospitalization)
        target_data = self.graph_snapshots[idx + self.history_window]
        y_target = target_data.y  # Shape: [num_nodes]
        
        return x_seq, y_seq, y_target

# Assuming we split the data into 70% training, 15% validation, 15% test
total_samples = len(graph_snapshots) - (history_window + pred_window) + 1
train_size = int(0.7 * total_samples)
val_size = int(0.15 * total_samples)
test_size = total_samples - train_size - val_size

# Create datasets
dataset = TemporalGraphDataset(graph_snapshots, history_window, pred_window)
train_dataset = torch.utils.data.Subset(dataset, range(train_size))
val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
test_dataset = torch.utils.data.Subset(dataset, range(train_size + val_size, total_samples))

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SpatioTemporalGNN(nn.Module):
    def __init__(self, num_features, hidden_dim, num_layers, pred_window, num_nodes):
        super(SpatioTemporalGNN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # GAT Layers for spatial modeling
        self.gat_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gat_layers.append(GATConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=4, concat=False))
        
        # GRU for temporal modeling
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        
        # Fully connected layer for output
        self.fc = nn.Linear(hidden_dim, pred_window)
        
        # Initial transformation layer
        self.input_fc = nn.Linear(num_features, hidden_dim)

    def forward(self, x_seq, edge_index, edge_weight):
        # x_seq: [batch_size, history_window, num_nodes, num_features]
        batch_size, seq_len, num_nodes, num_features = x_seq.size()
        
        # Reshape for processing
        x_seq = x_seq.view(batch_size * seq_len * num_nodes, num_features)
        x_seq = self.input_fc(x_seq)
        x_seq = x_seq.view(batch_size * seq_len, num_nodes, self.hidden_dim)
        
        # Apply GAT layers
        for gat in self.gat_layers:
            x_seq = F.elu(gat(x_seq.view(-1, self.hidden_dim), edge_index, edge_weight))
            x_seq = x_seq.view(batch_size * seq_len, num_nodes, self.hidden_dim)
        
        # Reshape for GRU
        x_seq = x_seq.view(batch_size, seq_len, num_nodes * self.hidden_dim)
        
        # Temporal modeling with GRU
        out, _ = self.gru(x_seq)
        
        # Output for the last time step
        out = out[:, -1, :]  # Shape: [batch_size, num_nodes * hidden_dim]
        
        # Predict hospitalization for next day
        out = self.fc(out)
        out = out.view(batch_size, num_nodes)
        
        return out

num_features = num_static_features + num_dynamic_features
hidden_dim = 64
num_layers = 5
pred_window = 1  # Predict next day's hospitalization
num_nodes = len(states)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SpatioTemporalGNN(num_features, hidden_dim, num_layers, pred_window, num_nodes)
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for x_seq, y_seq, y_target in train_loader:
        x_seq = x_seq.to(device)
        y_target = y_target.to(device)
        
        optimizer.zero_grad()
        
        # Since edge_index and edge_weight are the same for all snapshots, we can use them directly
        out = model(x_seq, edge_index.to(device), edge_weight.to(device))
        
        loss = criterion(out, y_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    # Validation loss
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x_seq, y_seq, y_target in val_loader:
            x_seq = x_seq.to(device)
            y_target = y_target.to(device)
            
            out = model(x_seq, edge_index.to(device), edge_weight.to(device))
            loss = criterion(out, y_target)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")



