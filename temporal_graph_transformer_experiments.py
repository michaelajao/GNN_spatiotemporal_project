"""
temporal_graph_transformer_experiments.py

Temporal Graph Transformer (TGT) Experiments for COVID Occupied MV Beds Forecasting
====================================================================================

This script conducts experiments using the Temporal Graph Transformer (TGT) model
to forecast COVID Occupied MV Beds across multiple regions. It includes:
1. Data loading and preprocessing.
2. Model definition and initialization.
3. Training, validation, and testing pipelines.
4. Evaluation metrics computation.
5. Visualization of results.

Author:
-------
[Your Name], [Institution or Group], [Year]

License:
--------
[Specify your license, e.g., MIT License]
"""

# ==============================================================================
# 0. Imports and Configuration
# ==============================================================================
import os
import random
import math
from math import radians, cos, sin, asin, sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import (mean_absolute_error,
                             mean_squared_error,
                             r2_score)
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates

# ------------------------------------------------------------------------------
# Set default style for publication-quality plots
# ------------------------------------------------------------------------------
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.size': 12,
    'figure.dpi': 300  # High resolution (DPI) for publication
})

# ==============================================================================
# 1. Random Seed & Device Configuration
# ==============================================================================
RANDOM_SEED = 123

def seed_torch(seed=RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Info] Using device: {device}")

# ==============================================================================
# 2. Hyperparameters
# ==============================================================================
num_nodes = 7
num_features = 5  # [new_confirmed, new_deceased, newAdmissions, hospitalCases, covidOccupiedMVBeds]
num_timesteps_input = 14
num_timesteps_output = 7

# Temporal Graph Transformer specific hyperparameters
d_model = 64
nhead = 4
num_layers = 2
dropout = 0.1

learning_rate = 1e-4
num_epochs = 1000
batch_size = 32
threshold_distance = 300  # km threshold for adjacency
early_stopping_patience = 20

# ==============================================================================
# 3. Reference Coordinates
# ==============================================================================
REFERENCE_COORDINATES = {
    "East of England": (52.1766, 0.425889),
    "Midlands": (52.7269, -1.458210),
    "London": (51.4923, -0.308660),
    "South East": (51.4341, -0.969570),
    "South West": (50.8112, -3.633430),
    "North West": (53.8981, -2.657550),
    "North East and Yorkshire": (54.5378, -2.180390),
}

# ==============================================================================
# 4. Data Loading and Preprocessing
# ==============================================================================
def load_and_correct_data(data: pd.DataFrame,
                          reference_coordinates: dict) -> pd.DataFrame:
    """
    Correct the latitude and longitude for each region, apply a 7-day rolling mean 
    to selected features, and sort the dataframe chronologically.
    """
    # Assign correct geographic coordinates
    for region, coords in reference_coordinates.items():
        data.loc[data['areaName'] == region, ['latitude', 'longitude']] = coords

    # Define features to apply rolling
    rolling_features = ['new_confirmed', 'new_deceased',
                        'newAdmissions', 'hospitalCases', 'covidOccupiedMVBeds']

    # 7-day rolling mean per region
    data[rolling_features] = (
        data.groupby('areaName')[rolling_features]
            .rolling(window=7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
    )

    # Fill any missing values with 0
    data[rolling_features] = data[rolling_features].fillna(0)

    # Sort by region and date
    data.sort_values(['areaName', 'date'], inplace=True)
    return data

class NHSRegionDataset(Dataset):
    """
    NHSRegionDataset for sliding-window time-series forecasting.

    - X: (T_in, m, F)
    - Y: (T_out, m) => only feature 4: 'covidOccupiedMVBeds'
    """

    def __init__(self, data: pd.DataFrame,
                 num_timesteps_input: int,
                 num_timesteps_output: int,
                 scaler: object = None):
        super().__init__()
        self.data = data.copy()
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output

        # Extract region info
        self.regions = self.data['areaName'].unique()
        self.num_nodes = len(self.regions)
        self.region_to_idx = {region: idx for idx, region in enumerate(self.regions)}
        self.data['region_idx'] = self.data['areaName'].map(self.region_to_idx)

        # Features to keep
        self.features = ['new_confirmed', 'new_deceased', 'newAdmissions',
                         'hospitalCases', 'covidOccupiedMVBeds']

        # Pivot to time-series
        self.pivot = self.data.pivot(index='date', columns='region_idx',
                                     values=self.features)
        self.pivot.ffill(inplace=True)
        self.pivot.fillna(0, inplace=True)

        self.num_features = len(self.features)
        self.num_dates = self.pivot.shape[0]
        self.feature_array = self.pivot.values.reshape(self.num_dates, self.num_nodes, self.num_features)

        # Check population consistency
        populations = self.data.groupby('areaName')['population'].unique()
        inconsistent_pop = populations[populations.apply(len) > 1]
        if not inconsistent_pop.empty:
            raise ValueError(f"Inconsistent population values in regions: {inconsistent_pop.index.tolist()}")

        # Optional scaling
        if scaler is not None:
            self.scaler = scaler
            self.feature_array = self.feature_array.reshape(-1, self.num_features)
            self.feature_array = self.scaler.transform(self.feature_array)
            self.feature_array = self.feature_array.reshape(self.num_dates, self.num_nodes, self.num_features)
        else:
            self.scaler = None

    def __len__(self) -> int:
        return self.num_dates - self.num_timesteps_input - self.num_timesteps_output + 1

    def __getitem__(self, idx: int):
        X = self.feature_array[idx : idx + self.num_timesteps_input]  # shape: (T_in, m, F)
        Y = self.feature_array[idx + self.num_timesteps_input : idx + self.num_timesteps_input + self.num_timesteps_output, :, 4]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def compute_geographic_adjacency(regions: list,
                                 latitudes: list,
                                 longitudes: list,
                                 threshold: float = threshold_distance) -> torch.Tensor:
    """
    Creates a binary adjacency matrix based on distance threshold using Haversine.
    """
    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Earth's radius in km
        return c * r

    num_nodes = len(regions)
    adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j:
                adj_matrix[i][j] = 1
            else:
                distance = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                if distance <= threshold:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1
    return torch.tensor(adj_matrix, dtype=torch.float32)

# ==============================================================================
# 5. Model Definitions
# ==============================================================================
class TemporalTransformerEncoder(nn.Module):
    """
    Transformer Encoder for temporal data.
    """
    def __init__(self, d_model=64, nhead=4, dim_feedforward=128, num_layers=2, dropout=0.1):
        super(TemporalTransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pos_embedding = nn.Parameter(torch.zeros(1000, d_model))  # Adjust max length as needed

    def forward(self, x):
        """
        x shape: (batch_size, time_steps, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pos_embedding[:seq_len, :]
        encoded = self.transformer_encoder(x)  # (batch_size, time_steps, d_model)
        return encoded

class TemporalGraphTransformer(nn.Module):
    """
    Temporal Graph Transformer for spatiotemporal forecasting.
    """
    def __init__(self, num_nodes, num_features, num_timesteps_input, num_timesteps_output,
                 d_model=64, nhead=4, num_layers=2, dropout=0.1, device='cpu'):
        super(TemporalGraphTransformer, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Linear(num_features * num_nodes, d_model)

        # Transformer Encoder
        self.transformer = TemporalTransformerEncoder(d_model, nhead, dim_feedforward=d_model*2,
                                                     num_layers=num_layers, dropout=dropout)

        # Graph Attention Layer
        self.graph_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout)

        # Output projection: map from d_model to num_timesteps_output per node
        self.output_proj = nn.Linear(d_model, num_timesteps_output)

        # Device
        self.device = device

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, X, adj=None):
        """
        X shape: (batch_size, T_in, num_nodes, num_features)
        adj shape: (num_nodes, num_nodes) or None
        """
        B, T, N, F = X.size()
        assert N == self.num_nodes and F == self.num_features, "Mismatch in number of nodes or features."

        # Flatten node and feature dimensions
        X_flat = X.view(B, T, N * F)  # (batch_size, T_in, num_nodes * num_features)

        # Input projection
        X_proj = self.input_proj(X_flat)  # (batch_size, T_in, d_model)

        # Transformer encoding
        encoded = self.transformer(X_proj)  # (batch_size, T_in, d_model)

        # Incorporate Graph Attention
        if adj is not None:
            # Reshape encoded to (batch_size * num_nodes, T_in, d_model)
            encoded = encoded.unsqueeze(2).repeat(1, 1, N, 1).view(B * N, T, self.d_model)  # (B*N, T_in, d_model)

            # Apply graph attention
            attn_output, _ = self.graph_attention(encoded, encoded, encoded)  # (B*N, T_in, d_model)

            # Aggregate over time steps (e.g., mean)
            attn_output = attn_output.mean(dim=1)  # (B*N, d_model)

            # Reshape back to (B, N, d_model)
            attn_output = attn_output.view(B, N, self.d_model)  # (B, N, d_model)
        else:
            # If no adjacency is provided, aggregate over time
            attn_output = encoded.mean(dim=1)  # (B, d_model)

        # Output projection
        if adj is not None:
            # Apply output projection per node
            output = self.output_proj(attn_output)  # (B, N, T_out)
        else:
            # Apply output projection globally
            output = self.output_proj(attn_output)  # (B, T_out)

        # Permute to (B, T_out, N)
        if adj is not None:
            output = output.permute(0, 2, 1)  # (B, T_out, N)
        else:
            output = output.permute(0, 2, 1)  # Adjust based on desired output

        return output

# ==============================================================================
# 6. Directory Setup and Data
# ==============================================================================
os.makedirs('figures', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

csv_path = "data/merged_nhs_covid_data.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"[Error] CSV file not found at {csv_path}")

data = pd.read_csv(csv_path, parse_dates=['date'])
data = load_and_correct_data(data, REFERENCE_COORDINATES)

initial_dataset = NHSRegionDataset(data,
                                   num_timesteps_input=num_timesteps_input,
                                   num_timesteps_output=num_timesteps_output,
                                   scaler=None)
print(f"[Info] Total samples in initial dataset: {len(initial_dataset)}")

total_len = len(initial_dataset)
train_size = int(0.7 * total_len)
val_size   = int(0.15 * total_len)
test_size  = total_len - train_size - val_size

# Temporal split: first 70% train, next 15% val, last 15% test
train_indices = list(range(0, train_size))
val_indices   = list(range(train_size, train_size + val_size))
test_indices  = list(range(train_size + val_size, total_len))

# Fit scaler on training data
scaler = StandardScaler()
train_features = []
for i in range(train_size):
    X, _ = initial_dataset[i]
    train_features.append(X.numpy())

train_features = np.concatenate(train_features, axis=0).reshape(-1, num_features)
scaler.fit(train_features)

scaled_dataset = NHSRegionDataset(data,
                                  num_timesteps_input=num_timesteps_input,
                                  num_timesteps_output=num_timesteps_output,
                                  scaler=scaler)
print(f"[Info] Total samples in scaled dataset: {len(scaled_dataset)}")

train_subset = Subset(scaled_dataset, train_indices)
val_subset   = Subset(scaled_dataset, val_indices)
test_subset  = Subset(scaled_dataset, test_indices)

print(f"[Info] Training samples:   {len(train_subset)}")
print(f"[Info] Validation samples: {len(val_subset)}")
print(f"[Info] Test samples:       {len(test_subset)}")

train_loader = DataLoader(train_subset, batch_size=batch_size,
                          shuffle=False, drop_last=True)
val_loader   = DataLoader(val_subset,   batch_size=batch_size,
                          shuffle=False, drop_last=False)
test_loader  = DataLoader(test_subset,  batch_size=batch_size,
                          shuffle=False, drop_last=False)

# Adjacency matrix (Optional for TGT; can be set to None)
regions = scaled_dataset.regions.tolist()
latitudes  = [data[data['areaName'] == region]['latitude'].iloc[0]  for region in regions]
longitudes = [data[data['areaName'] == region]['longitude'].iloc[0] for region in regions]

adj_static = compute_geographic_adjacency(regions, latitudes, longitudes).to(device)
print("[Info] Static Adjacency Matrix:")
print(adj_static.cpu().numpy())

# Optional: visualize static adjacency as a geographic graph
adj_np = adj_static.cpu().numpy()
G = nx.from_numpy_array(adj_np)
mapping = {i: region for i, region in enumerate(regions)}
G = nx.relabel_nodes(G, mapping)
pos = {region: (longitudes[i], latitudes[i]) for i, region in enumerate(regions)}

plt.figure(figsize=(12, 10))
nx.draw_networkx(G, pos,
                 with_labels=True,
                 node_size=1000,
                 node_color='lightblue',
                 edge_color='gray',
                 font_size=12,
                 font_weight='bold')
plt.title('Geographic Adjacency Graph - Static Adjacency')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.axis('off')
plt.tight_layout()
plt.savefig('figures/geographic_adjacency_graph_static.png', dpi=300)
plt.show()

# ==============================================================================
# 7. Initialize Model
# ==============================================================================
def initialize_model(model_type='tgt'):
    """
    Initialize the Temporal Graph Transformer model.
    """
    if model_type.lower() == 'tgt':
        model = TemporalGraphTransformer(
            num_nodes=num_nodes,
            num_features=num_features,
            num_timesteps_input=num_timesteps_input,
            num_timesteps_output=num_timesteps_output,
            d_model=hyperparams['d_model'],
            nhead=hyperparams['nhead'],
            num_layers=hyperparams['num_layers'],
            dropout=hyperparams['dropout'],
            device=device
        ).to(device)
    else:
        raise ValueError("Unsupported model_type. Currently, only 'tgt' is implemented.")
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=3,
                                                     verbose=True)
    return model, optimizer, criterion, scheduler

# ==============================================================================
# 8. Training & Evaluation Function
# ==============================================================================
def run_experiment(model_type='tgt', experiment_id=1, summary_metrics=[]):
    """
    Run a full training/validation/testing cycle for the Temporal Graph Transformer (TGT).
    """
    print(f"\n[Experiment {experiment_id}] Starting with {model_type.upper()} Model...\n")

    model, optimizer, criterion, scheduler = initialize_model(model_type=model_type)

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()

            # Forward pass
            pred = model(batch_X, adj=adj_static)  # pred shape: (B, T_out, N)
            loss = criterion(pred, batch_Y)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        all_val_preds = []
        all_val_actuals = []

        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

                pred = model(batch_X, adj=adj_static)
                vloss = criterion(pred, batch_Y)
                epoch_val_loss += vloss.item()

                all_val_preds.append(pred.cpu().numpy())
                all_val_actuals.append(batch_Y.cpu().numpy())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # R² metrics
        all_val_preds = np.concatenate(all_val_preds, axis=0)
        all_val_actuals = np.concatenate(all_val_actuals, axis=0)
        preds_2d = all_val_preds.reshape(-1, num_nodes)
        actuals_2d = all_val_actuals.reshape(-1, num_nodes)
        r2_vals = r2_score(actuals_2d, preds_2d, multioutput='raw_values')

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} | Val R² (per node): {r2_vals}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_dir = f'models/experiment{experiment_id}'
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = f'{checkpoint_dir}/best_model.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f"[Info] Model checkpoint saved at {checkpoint_path}.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("[Info] Early stopping triggered.")
                break

    # Plot training vs. validation loss
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, len(train_losses) + 1), y=train_losses,
                 label='Train Loss', color='blue', marker='o')
    sns.lineplot(x=range(1, len(val_losses) + 1), y=val_losses,
                 label='Validation Loss', color='orange', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Training vs. Validation Loss - {model_type.upper()} Model')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = f'figures/training_validation_loss_experiment{experiment_id}_{model_type}.png'
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"[Info] Loss curves saved to {plot_path}")

    # Load best model for testing
    model.load_state_dict(torch.load(f'models/experiment{experiment_id}/best_model.pth', map_location=device))
    model.eval()

    # Testing
    test_loss = 0.0
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            pred = model(batch_X, adj=adj_static)
            loss = criterion(pred, batch_Y)
            test_loss += loss.item()

            all_preds.append(pred.cpu())
            all_actuals.append(batch_Y.cpu())

    avg_test_loss = test_loss / len(test_loader)
    print(f"[Experiment {experiment_id}] Test Loss (MSE): {avg_test_loss:.4f}")

    # Reshape final predictions
    all_preds = torch.cat(all_preds, dim=0)
    all_actuals = torch.cat(all_actuals, dim=0)

    # Inverse transform
    if scaled_dataset.scaler is not None:
        scale_covid = scaled_dataset.scaler.scale_[4]  # 'covidOccupiedMVBeds' index
        mean_covid  = scaled_dataset.scaler.mean_[4]
        all_preds_np   = all_preds.numpy()   * scale_covid + mean_covid
        all_actuals_np = all_actuals.numpy() * scale_covid + mean_covid
    else:
        all_preds_np   = all_preds.numpy()
        all_actuals_np = all_actuals.numpy()

    # Flatten
    preds_flat   = all_preds_np.reshape(-1, num_nodes)
    actuals_flat = all_actuals_np.reshape(-1, num_nodes)

    # Compute metrics
    mae_per_node = mean_absolute_error(actuals_flat, preds_flat, multioutput='raw_values')
    mse_per_node = mean_squared_error(actuals_flat, preds_flat, multioutput='raw_values')
    r2_per_node  = r2_score(actuals_flat, preds_flat, multioutput='raw_values')

    pearson_per_node = []
    for i in range(num_nodes):
        # Avoid division by zero
        if np.std(preds_flat[:, i]) == 0 or np.std(actuals_flat[:, i]) == 0:
            pearson_cc = 0
        else:
            pearson_cc, _ = pearsonr(preds_flat[:, i], actuals_flat[:, i])
        pearson_per_node.append(pearson_cc)

    # Organize metrics into a CSV
    metrics_dict = {
        'Experiment_ID': [],
        'Model_Type': [],
        'Region': [],
        'MAE': [],
        'MSE': [],
        'R2_Score': [],
        'Pearson_CC': []
    }

    for idx, region in enumerate(regions):
        metrics_dict['Experiment_ID'].append(experiment_id)
        metrics_dict['Model_Type'].append(model_type.upper())
        metrics_dict['Region'].append(region)
        metrics_dict['MAE'].append(mae_per_node[idx])
        metrics_dict['MSE'].append(mse_per_node[idx])
        metrics_dict['R2_Score'].append(r2_per_node[idx])
        metrics_dict['Pearson_CC'].append(pearson_per_node[idx])

        summary_metrics.append({
            'Experiment_ID': experiment_id,
            'Model_Type': model_type.upper(),
            'Region': region,
            'MAE': mae_per_node[idx],
            'MSE': mse_per_node[idx],
            'R2_Score': r2_per_node[idx],
            'Pearson_CC': pearson_per_node[idx]
        })

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_csv_path = f'results/metrics/metrics_experiment{experiment_id}_{model_type}.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[Info] Metrics saved to {metrics_csv_path}")

    # Visualization steps
    unique_dates = data['date'].sort_values().unique()
    forecast_dates = []
    num_test_samples = len(test_subset)

    for i in range(num_test_samples):
        pred_start_idx = train_size + val_size + i + num_timesteps_input
        pred_end_idx   = pred_start_idx + num_timesteps_output
        if pred_end_idx > len(unique_dates):
            pred_end_idx = len(unique_dates)
        sample_dates = unique_dates[pred_start_idx : pred_end_idx]
        if len(sample_dates) < num_timesteps_output:
            last_date = unique_dates[-1]
            sample_dates = np.append(sample_dates, [last_date] * (num_timesteps_output - len(sample_dates)))
        forecast_dates.extend(sample_dates)

    # Build DataFrames
    preds_df = pd.DataFrame(all_preds_np.reshape(-1, num_nodes), columns=regions)
    preds_df['Date'] = forecast_dates

    actuals_df = pd.DataFrame(all_actuals_np.reshape(-1, num_nodes), columns=regions)
    actuals_df['Date'] = forecast_dates

    agg_preds_df  = preds_df.groupby('Date').mean().reset_index()
    agg_actuals_df= actuals_df.groupby('Date').first().reset_index()

    merged_df = pd.merge(agg_preds_df, agg_actuals_df, on='Date', suffixes=('_Predicted', '_Actual'))
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])

    # Time-series Plots per Region
    def plot_time_series(region, df, model_type, experiment_id):
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Date', y=f'{region}_Actual', data=df,
                     label='Actual', color='blue', marker='o')
        sns.lineplot(x='Date', y=f'{region}_Predicted', data=df,
                     label='Predicted', color='red', linestyle='--', marker='x')
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.title(f'Actual vs Predicted COVID Occupied MV Beds - {region} ({model_type.upper()})')
        plt.xlabel('Date')
        plt.ylabel('COVID Occupied MV Beds')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = f'figures/actual_vs_predicted_experiment{experiment_id}_{model_type}_{region.replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
        print(f"[Info] Time series plot saved to {plot_path}")

    for region in regions:
        plot_time_series(region, merged_df, model_type, experiment_id)

    # Overall Metrics Plot
    def plot_overall_metrics(summary_metrics, experiment_id, model_type):
        """
        Aggregates and plots overall metrics (MAE, MSE, R2, PCC) across all regions.
        """
        df = pd.DataFrame(summary_metrics)
        df = df[(df['Experiment_ID'] == experiment_id) & 
                (df['Model_Type'] == model_type.upper())]
        aggregated_metrics = df[['MAE','MSE','R2_Score','Pearson_CC']].mean()

        plt.figure(figsize=(8, 6))
        sns.barplot(x=aggregated_metrics.index, y=aggregated_metrics.values, palette='Set2')
        plt.title(f'Overall Metrics (Averaged) - Experiment {experiment_id} ({model_type.upper()})')
        plt.ylabel('Metric Value')
        plt.ylim(0, max(aggregated_metrics.values) * 1.2)
        for i, v in enumerate(aggregated_metrics.values):
            plt.text(i, v + 0.01*max(aggregated_metrics.values), f"{v:.4f}",
                     ha='center', va='bottom', fontsize=12)
        plt.tight_layout()
        plot_path = f'figures/overall_metrics_experiment{experiment_id}_{model_type}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
        print(f"[Info] Overall metrics bar chart saved to {plot_path}")

    plot_overall_metrics(summary_metrics, experiment_id, model_type)

    # Boxplot of Errors
    def plot_error_boxplot(df, model_type, experiment_id):
        error_df = df.copy()
        for region in regions:
            error_df[f'{region}_Error'] = error_df[f'{region}_Predicted'] - error_df[f'{region}_Actual']

        # Melt
        var_cols = [f'{region}_Error' for region in regions]
        error_melted = error_df.melt(id_vars=['Date'], value_vars=var_cols,
                                     var_name='Region', value_name='Error')
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Region', y='Error', data=error_melted, palette='Set3')
        sns.swarmplot(x='Region', y='Error', data=error_melted, color=".25", alpha=0.6)
        plt.title(f'Prediction Errors Boxplot ({model_type.upper()})')
        plt.xlabel('Region')
        plt.ylabel('Prediction Error (Pred - Actual)')
        plt.tight_layout()
        plot_path = f'figures/boxplot_errors_experiment{experiment_id}_{model_type}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
        print(f"[Info] Boxplot of errors saved to {plot_path}")

    plot_error_boxplot(merged_df, model_type, experiment_id)

    # Scatter Actual vs Predicted (Overall)
    def plot_scatter_actual_vs_predicted_overall(df, model_type, experiment_id):
        # Combine columns for overall
        plt.figure(figsize=(7, 7))
        df_overall = df.copy()
        df_overall['covidOccupiedMVBeds_Actual'] = df_overall[[c for c in df_overall.columns if c.endswith('_Actual')]].mean(axis=1)
        df_overall['covidOccupiedMVBeds_Predicted'] = df_overall[[c for c in df_overall.columns if c.endswith('_Predicted')]].mean(axis=1)

        sns.scatterplot(x=df_overall['covidOccupiedMVBeds_Actual'],
                        y=df_overall['covidOccupiedMVBeds_Predicted'],
                        color='teal', alpha=0.6)
        sns.regplot(x=df_overall['covidOccupiedMVBeds_Actual'],
                    y=df_overall['covidOccupiedMVBeds_Predicted'],
                    scatter=False, color='red', label='Regression Line')
        plt.title(f'Overall Actual vs Predicted (Avg) ({model_type.upper()})')
        plt.xlabel('Actual COVID Occupied MV Beds')
        plt.ylabel('Predicted COVID Occupied MV Beds')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = f'figures/scatter_actual_vs_predicted_overall_experiment{experiment_id}_{model_type}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
        print(f"[Info] Overall scatter plot saved to {plot_path}")

    plot_scatter_actual_vs_predicted_overall(merged_df, model_type, experiment_id)

    # Save final model
    final_model_path = f'models/experiment{experiment_id}/{model_type}_final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"[Info] Final model saved as '{final_model_path}'.")

    print(f"[Experiment {experiment_id}] Workflow complete.\n")

# ==============================================================================
# 9. Main: Run All Experiments
# ==============================================================================
if __name__ == "__main__":
    experiments = [
        {'model_type': 'tgt',    'experiment_id': 1},
    ]

    summary_metrics = []

    # Execute each experiment
    for exp in experiments:
        run_experiment(model_type=exp['model_type'],
                       experiment_id=exp['experiment_id'],
                       summary_metrics=summary_metrics)

    # Summarize results
    summary_df = pd.DataFrame(summary_metrics)
    summary_csv = 'results/metrics/summary_metrics.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Info] Summary metrics saved to {summary_csv}")

    # Pivot to show average metrics per experiment
    summary_pivot = summary_df.groupby(['Experiment_ID', 'Model_Type']).agg({
        'MAE': 'mean',
        'MSE': 'mean',
        'R2_Score': 'mean',
        'Pearson_CC': 'mean'
    }).reset_index()

    print("\nSummary of All Experiments:\n", summary_pivot)

    summary_pivot_path = 'results/metrics/summary_metrics_pivot.csv'
    summary_pivot.to_csv(summary_pivot_path, index=False)
    print(f"[Info] Summary pivot table saved to {summary_pivot_path}")

    # Optional comparison plot for summary metrics
    def plot_summary_metrics(summary_pivot):
        metrics_list = ['MAE', 'MSE', 'R2_Score', 'Pearson_CC']
        num_metrics = len(metrics_list)
        plt.figure(figsize=(15, 10))
        for i, metric in enumerate(metrics_list):
            plt.subplot(2, 2, i+1)
            sns.barplot(x='Model_Type', y=metric, data=summary_pivot, hue='Model_Type', palette='Set2', dodge=True)
            plt.title(f'Average {metric}')
            plt.ylabel(metric)
            plt.xlabel('Model Type')
            plt.legend(title='Model Type')
            for j, row in summary_pivot.iterrows():
                plt.text(j % len(summary_pivot), row[metric] + 0.01*row[metric], f"{row[metric]:.4f}",
                         ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        summary_plot = 'figures/summary_metrics_comparison.png'
        plt.savefig(summary_plot, dpi=300)
        plt.show()
        print(f"[Info] Summary metrics comparison plot saved to {summary_plot}")

    plot_summary_metrics(summary_pivot)
