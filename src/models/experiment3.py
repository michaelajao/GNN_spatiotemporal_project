import os
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from haversine import haversine
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from epiweeks import Week
from sklearn.preprocessing import StandardScaler
from dtw import accelerated_dtw
from sklearn.model_selection import train_test_split

# =======================
# Step 1: Set Parameters
# =======================

K_NEIGHBORS = 5  # Number of nearest neighbors for each state
START_DATE = '2020-06-01'
END_DATE = '2021-12-01'
TARGET_FEATURE = 'hospitalization'
TIME_WINDOWS = [3, 7]  # Time windows to experiment with

# =======================
# Step 2: Load and Preprocess Data
# =======================

# 2.1 Load COVID-19 Data
covid_data_path = '../../data/processed/processed_covid_data.pickle'
with open(covid_data_path, 'rb') as f:
    raw_data = pickle.load(f)

# 2.2 Load Population Data
pop_data_path = '../../data/uszips.csv'
pop_data = pd.read_csv(pop_data_path)

# 2.3 Aggregate Population Data by State
pop_data = pop_data.groupby('state_name').agg({
    'population': 'sum',
    'density': 'mean',
    'lat': 'mean',
    'lng': 'mean'
}).reset_index()

# 2.4 Merge COVID Data with Population Data
raw_data = pd.merge(
    raw_data,
    pop_data,
    how='inner',
    left_on='state',
    right_on='state_name'
)

# 2.5 Convert 'date_today' Column to Datetime
raw_data['date_today'] = pd.to_datetime(raw_data['date_today'])

# 2.6 Filter Data Between START_DATE and END_DATE
raw_data = raw_data[
    (raw_data['date_today'] >= pd.to_datetime(START_DATE)) &
    (raw_data['date_today'] <= pd.to_datetime(END_DATE))
].reset_index(drop=True)

# 2.7 Handle Missing Values in 'hospitalization' Column
raw_data['hospitalization'] = raw_data['hospitalization'].fillna(0)

# 2.8 Map Dates to Weeks
raw_data['week'] = raw_data['date_today'].dt.to_period('W').apply(lambda r: r.start_time)

# 2.9 Aggregate Data by Week and State
weekly_data = raw_data.groupby(['week', 'state']).agg({
    'hospitalization': 'mean',
    'population': 'first',
    'density': 'first',
    'lat': 'first',
    'lng': 'first'
}).reset_index()

# =======================
# Step 3: Data Normalization
# =======================

# 3.1 Initialize Scalers
hospitalization_scaler = StandardScaler()

# 3.2 Fit Scaler on Training Data
states = weekly_data['state'].unique()
state_scalers = {}
for state in states:
    state_data = weekly_data[weekly_data['state'] == state]
    scaler = StandardScaler()
    scaler.fit(state_data[[TARGET_FEATURE]])
    state_scalers[state] = scaler
    weekly_data.loc[weekly_data['state'] == state, TARGET_FEATURE] = scaler.transform(state_data[[TARGET_FEATURE]])

# =======================
# Step 4: Create Time Series Data
# =======================

# 4.1 Prepare Sequences with Different Time Windows
sequence_data = {}
for window_size in TIME_WINDOWS:
    sequences = []
    targets = []
    for state in states:
        state_data = weekly_data[weekly_data['state'] == state].sort_values('week')
        state_features = state_data[[TARGET_FEATURE]].values
        for i in range(len(state_features) - window_size):
            seq_x = state_features[i:i+window_size]
            seq_y = state_features[i+window_size]
            sequences.append((state, seq_x))
            targets.append(seq_y)
    sequence_data[window_size] = (sequences, targets)

# =======================
# Step 5: Split Data into Train, Validation, and Test Sets
# =======================

train_data = {}
val_data = {}
test_data = {}
for window_size in TIME_WINDOWS:
    sequences, targets = sequence_data[window_size]
    train_seqs, test_seqs, train_tgt, test_tgt = train_test_split(
        sequences, targets, test_size=0.2, shuffle=False
    )
    train_seqs, val_seqs, train_tgt, val_tgt = train_test_split(
        train_seqs, train_tgt, test_size=0.1, shuffle=False
    )
    train_data[window_size] = (train_seqs, train_tgt)
    val_data[window_size] = (val_seqs, val_tgt)
    test_data[window_size] = (test_seqs, test_tgt)

# =======================
# Step 6: Construct Graphs Using Dynamic Time Warping (DTW)
# =======================

state_time_series = {}
for state in states:
    state_data = weekly_data[weekly_data['state'] == state].sort_values('week')
    state_time_series[state] = state_data[TARGET_FEATURE].values

# 6.1 Compute DTW Distances Between States
dtw_distance_matrix = np.zeros((len(states), len(states)))
for i, state_i in enumerate(states):
    for j, state_j in enumerate(states):
        if i <= j:
            x = state_time_series[state_i]
            y = state_time_series[state_j]
            distance, _, _, _ = accelerated_dtw(x, y, dist='euclidean')
            dtw_distance_matrix[i, j] = distance
            dtw_distance_matrix[j, i] = distance

# 6.2 Construct K-Nearest Neighbors Graph Based on DTW Distances
neighbors = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric='precomputed')
neighbors.fit(dtw_distance_matrix)
distances, indices = neighbors.kneighbors(dtw_distance_matrix)

# 6.3 Create Edge Index and Edge Weight Tensors
edge_index = []
edge_weight = []
for i in range(len(states)):
    for j in range(1, K_NEIGHBORS + 1):  # Start from 1 to exclude self-loop
        neighbor_idx = indices[i, j]
        edge_index.append([i, neighbor_idx])
        # Edge weight inversely proportional to DTW distance
        weight = 1 / (dtw_distance_matrix[i, neighbor_idx] + 1e-5)
        edge_weight.append(weight)

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
edge_weight = torch.tensor(edge_weight, dtype=torch.float)

# Make the graph undirected
edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)
edge_weight = torch.cat([edge_weight, edge_weight], dim=0)

# =======================
# Step 7: Prepare Data for GNN Model
# =======================

# 7.1 Create State Mapping
state_to_idx = {state: idx for idx, state in enumerate(states)}

# 7.2 Prepare Data Loaders
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets, state_to_idx):
        self.sequences = sequences
        self.targets = targets
        self.state_to_idx = state_to_idx

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        state, seq = self.sequences[idx]
        state_idx = self.state_to_idx[state]
        seq = torch.tensor(seq, dtype=torch.float)
        target = torch.tensor(self.targets[idx], dtype=torch.float)
        return state_idx, seq, target

# 7.3 Create Data Loaders for Each Time Window
batch_size = 64
train_loaders = {}
val_loaders = {}
test_loaders = {}

for window_size in TIME_WINDOWS:
    train_seqs, train_tgt = train_data[window_size]
    val_seqs, val_tgt = val_data[window_size]
    test_seqs, test_tgt = test_data[window_size]

    train_dataset = TimeSeriesDataset(train_seqs, train_tgt, state_to_idx)
    val_dataset = TimeSeriesDataset(val_seqs, val_tgt, state_to_idx)
    test_dataset = TimeSeriesDataset(test_seqs, test_tgt, state_to_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    train_loaders[window_size] = train_loader
    val_loaders[window_size] = val_loader
    test_loaders[window_size] = test_loader

# =======================
# Step 8: Define the GNN Model
# =======================

import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN_GRU_Model(nn.Module):
    def __init__(self, num_states, window_size, edge_index, edge_weight):
        super(GCN_GRU_Model, self).__init__()
        self.num_states = num_states
        self.window_size = window_size
        self.edge_index = edge_index
        self.edge_weight = edge_weight

        self.gcn1 = GCNConv(window_size, 32)
        self.gcn2 = GCNConv(32, 16)

        self.gru = nn.GRU(16, 16, batch_first=True)

        self.fc = nn.Linear(16, 1)

    def forward(self, state_indices, sequences):
        # Prepare node features
        x = torch.zeros(self.num_states, self.window_size, device=sequences.device)
        x[state_indices] = sequences

        # First GCN layer
        x = self.gcn1(x, self.edge_index, self.edge_weight)
        x = F.relu(x)

        # Second GCN layer
        x = self.gcn2(x, self.edge_index, self.edge_weight)
        x = F.relu(x)

        # Extract node features for the batch states
        x = x[state_indices]

        # GRU layer
        x = x.unsqueeze(1)  # Add sequence dimension
        _, h_n = self.gru(x)
        h_n = h_n.squeeze(0)

        # Fully connected layer
        out = self.fc(h_n)

        return out

# =======================
# Step 9: Train the Model
# =======================

from torch.optim import Adam

# Training parameters
num_epochs = 50
learning_rate = 0.001

for window_size in TIME_WINDOWS:
    print(f"Training for window size: {window_size}")

    num_states = len(states)
    model = GCN_GRU_Model(num_states, window_size, edge_index, edge_weight)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    train_loader = train_loaders[window_size]
    val_loader = val_loaders[window_size]

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for state_indices, sequences, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(state_indices, sequences)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * sequences.size(0)

        avg_loss = total_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for state_indices, sequences, targets in val_loader:
                outputs = model(state_indices, sequences)
                loss = criterion(outputs.squeeze(), targets)
                val_loss += loss.item() * sequences.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

# =======================
# Step 10: Evaluate the Model
# =======================

from sklearn.metrics import mean_squared_error

for window_size in TIME_WINDOWS:
    print(f"Testing for window size: {window_size}")

    model.eval()
    test_loader = test_loaders[window_size]
    predictions = []
    actuals = []

    with torch.no_grad():
        for state_indices, sequences, targets in test_loader:
            outputs = model(state_indices, sequences)
            predictions.extend(outputs.squeeze().tolist())
            actuals.extend(targets.tolist())

    # Inverse transform the predictions and actuals
    all_states = [seq[0] for seq in test_data[window_size][0]]
    inv_predictions = []
    inv_actuals = []
    for i in range(len(predictions)):
        state = all_states[i]
        scaler = state_scalers[state]
        inv_pred = scaler.inverse_transform([[predictions[i]]])[0][0]
        inv_actual = scaler.inverse_transform([[actuals[i]]])[0][0]
        inv_predictions.append(inv_pred)
        inv_actuals.append(inv_actual)

    rmse = np.sqrt(mean_squared_error(inv_actuals, inv_predictions))
    print(f"Test RMSE for window size {window_size}: {rmse:.4f}")

# =======================
# Step 11: Visualize Results
# =======================

import matplotlib.pyplot as plt

# Example: Plot predictions vs actuals for a specific state
state_to_plot = 'California'
indices = [i for i, seq in enumerate(test_data[window_size][0]) if seq[0] == state_to_plot]

state_preds = [inv_predictions[i] for i in indices]
state_actuals = [inv_actuals[i] for i in indices]
weeks = weekly_data['week'].unique()[-len(state_preds):]

plt.figure(figsize=(12, 6))
plt.plot(weeks, state_actuals, label='Actual')
plt.plot(weeks, state_preds, label='Predicted')
plt.title(f'Hospitalization Prediction for {state_to_plot}')
plt.xlabel('Week')
plt.ylabel('Hospitalization')
plt.legend()
plt.show()
