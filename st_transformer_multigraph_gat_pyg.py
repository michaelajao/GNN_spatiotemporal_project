#!/usr/bin/env python
# coding: utf-8

"""
st_transformer_multigraph_gat_pyg.py

Spatiotemporal Transformer + Graph Attention Networks (GAT) using PyTorch Geometric
====================================================================================
This script implements an enhanced spatiotemporal forecasting model that:
1. Uses MinMax scaling for data normalization (applied to the target variable).
2. Incorporates multiple forecast horizons (3, 7, 14 days).
3. Computes and saves metrics (MAE, MSE, RMSE, Pearson_CC) for each horizon and region.
4. Plots and saves training vs validation loss and summary metrics.
   (Excludes forecast vs actual plots for research paper use.)

The model architecture includes:
- A Transformer-based temporal encoder for capturing global temporal dependencies.
- Multi-graph adjacency fusion (geographic, correlation, dynamic).
- A Graph Attention Network (GAT) stack using PyTorch Geometric for spatial dependencies.
- Output layers for multiple forecast horizons.

Author: Your Name
Date: 2025-01-24
License: MIT
"""

# ==============================================================================
# 0. Imports & Configuration
# ==============================================================================
import os
import math
import random
from math import radians, sin, cos, asin, sqrt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F  # Ensure this is not shadowed
import torch.optim as optim
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler

# PyTorch Geometric Imports
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeometricDataLoader

# Matplotlib style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.size': 12,
    'figure.dpi': 300
})

# Set random seed for reproducibility
RANDOM_SEED = 123

def seed_torch(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Info] Using device: {device}")

# Hyperparameters
num_features = 5  # [new_confirmed, new_deceased, newAdmissions, hospitalCases, covidOccupiedMVBeds]
num_timesteps_input = 14
num_timesteps_output = 14  # Max forecast horizon
learning_rate = 1e-4
batch_size = 32
num_epochs = 300
early_stopping_patience = 20
threshold_distance = 300.0  # for geographic adjacency

# Forecast horizons to evaluate
forecast_horizons = [3, 7, 14]

# Directories
os.makedirs('figures', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

# ==============================================================================
# 1. Data Preprocessing & Adjacency
# ==============================================================================
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth surface.
    """
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Earth radius in kilometers
    return c * r

def get_batched_edge_index(adj, batch_size):
    """
    Convert a batch of adjacency matrices to batched edge indices.

    Args:
        adj: Tensor of shape [B, m, m], binary adjacency matrices.
        batch_size: int, number of graphs in the batch.

    Returns:
        edge_index: Tensor of shape [2, E_total], concatenated edge indices.
        batch: Tensor of shape [B * m], indicating graph index for each node.
    """
    edge_indices = []
    for b in range(batch_size):
        adj_b = adj[b]  # [m, m]
        src, dst = adj_b.nonzero(as_tuple=True)  # Indices where adj=1
        src = src + b * adj.size(1)  # Offset for batching
        dst = dst + b * adj.size(1)
        if len(src) > 0:
            edge_indices.append(torch.stack([src, dst], dim=0))
    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1).long()  # [2, E_total]
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long).to(adj.device)
    # Create batch vector
    num_nodes = adj.size(1)
    batch = torch.arange(batch_size).repeat_interleave(num_nodes).to(adj.device)  # [B * m]
    return edge_index, batch

def load_and_compute_adjacencies(
    data: pd.DataFrame,
    reference_coordinates: dict,
    correlation_feature='covidOccupiedMVBeds',
    corr_window=None
):
    """
    1) Apply 7-day rolling means on relevant features,
    2) Compute pivot for correlation adjacency on `correlation_feature`,
    3) Build correlation adjacency A_corr,
    4) Build geographic adjacency A_geo,
    5) Return the processed DataFrame and adjacency tensors.
    """

    # A) Assign lat/lon
    for region, coords in reference_coordinates.items():
        data.loc[data['areaName'] == region, ['latitude','longitude']] = coords

    # B) 7-day rolling means
    rolling_feats = ['new_confirmed','new_deceased','newAdmissions','hospitalCases','covidOccupiedMVBeds']
    data[rolling_feats] = (
        data.groupby('areaName')[rolling_feats]
            .rolling(window=7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
    )
    data[rolling_feats] = data[rolling_feats].fillna(0)
    data.sort_values(['areaName','date'], inplace=True)

    # C) Build correlation pivot
    pivot_df = data.pivot(index='date', columns='areaName', values=correlation_feature)
    pivot_df = pivot_df.fillna(0.0)
    pivot_df.sort_index(inplace=True)

    if corr_window is not None and corr_window < len(pivot_df):
        pivot_df = pivot_df.iloc[-corr_window:]  # last corr_window days

    regions = pivot_df.columns.tolist()
    m = len(regions)

    # 1) Geographic adjacency
    latitudes = []
    longitudes = []
    for reg in regions:
        lat, lon = reference_coordinates[reg]
        latitudes.append(lat)
        longitudes.append(lon)

    A_geo_np = np.zeros((m, m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            if i == j:
                A_geo_np[i, j] = 1.
            else:
                dist = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                if dist <= threshold_distance:
                    A_geo_np[i, j] = 1.
                    A_geo_np[j, i] = 1.
    A_geo = torch.tensor(A_geo_np, dtype=torch.float32)

    # 2) Correlation adjacency
    data_for_corr = pivot_df.values.T  # shape [m, T]
    corr_matrix = np.corrcoef(data_for_corr)  # shape [m, m]
    np.fill_diagonal(corr_matrix, 1.0)
    A_corr_np = corr_matrix.astype(np.float32)
    A_corr = torch.tensor(A_corr_np, dtype=torch.float32)

    return data, A_geo, A_corr

# ==============================================================================
# 2. Dataset Definitions
# ==============================================================================
class MyCovidDataset(Dataset):
    """
    Dataset class for COVID-19 data.
    Provides input sequences and corresponding targets.
    Supports multiple forecast horizons.
    """

    def __init__(self, data, num_timesteps_input=14, num_timesteps_output=14, scaler=None):
        super().__init__()
        self.data = data.copy()
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output

        # Region ordering
        self.regions = data['areaName'].unique()
        self.num_nodes = len(self.regions)
        self.region_to_idx = {r: i for i, r in enumerate(self.regions)}
        self.data['region_idx'] = self.data['areaName'].map(self.region_to_idx)

        # Features
        self.features = ['new_confirmed','new_deceased','newAdmissions','hospitalCases','covidOccupiedMVBeds']

        # Pivot => shape: [date, region_idx, feats]
        pivot_df = self.data.pivot(index='date', columns='region_idx', values=self.features)
        pivot_df = pivot_df.fillna(0.0)
        pivot_df.sort_index(inplace=True)

        self.num_features = len(self.features)
        self.num_dates = pivot_df.shape[0]
        self.feature_array = pivot_df.values.reshape(self.num_dates, self.num_nodes, self.num_features)

        if scaler is not None:
            self.scaler = scaler
            # Scale only the target feature ('covidOccupiedMVBeds')
            self.feature_array[:,:,4] = self.scaler.fit_transform(
                self.feature_array[:,:,4].reshape(-1, 1)
            ).reshape(self.feature_array[:,:,4].shape)
        else:
            self.scaler = None

    def __len__(self):
        return self.num_dates - self.num_timesteps_input - self.num_timesteps_output + 1

    def __getitem__(self, idx):
        X = self.feature_array[idx : idx + self.num_timesteps_input]  # (T_in, m, F)
        Y = self.feature_array[idx + self.num_timesteps_input : idx + self.num_timesteps_input + self.num_timesteps_output, :, 4]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# ==============================================================================
# 3. Model Definitions
# ==============================================================================
class TransformerTemporalEncoder(nn.Module):
    """
    Transformer-based temporal encoder for capturing global temporal dependencies.
    """
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True  # Set batch_first=True to align with data
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Project input features to d_model
        self.input_proj = nn.Linear(5, d_model)

    def forward(self, X):
        """
        X => [B, T_in, m, C]
        Output => [B, m, d_model]
        """
        B, T_in, m, C = X.shape
        X_reshaped = X.permute(0,2,1,3)  # => [B, m, T_in, C]
        Bm = B * m
        X_reshaped = X_reshaped.reshape(Bm, T_in, C)  # => [Bm, T_in, C]
        X_proj = self.input_proj(X_reshaped)          # => [Bm, T_in, d_model]
        # No need to permute for batch_first=True
        out_tf = self.transformer_encoder(X_proj)     # => [Bm, T_in, d_model]
        out_pool = torch.mean(out_tf, dim=1)          # [Bm, d_model]
        out_final = out_pool.reshape(B, m, self.d_model)
        return out_final

class GraphLearner(nn.Module):
    """
    Learns dynamic adjacency matrices from node embeddings.
    """
    def __init__(self, hidden_dim, alpha=1.0):
        super().__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.alpha = alpha

    def forward(self, emb):
        """
        emb => [B, m, hidR]
        returns => [B, m, m]
        """
        B, m, hd = emb.shape
        x1 = torch.tanh(self.alpha * self.lin1(emb))
        x2 = torch.tanh(self.alpha * self.lin2(emb))
        adj = torch.bmm(x1, x2.transpose(1,2)) - torch.bmm(x2, x1.transpose(1,2))
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))
        return adj

class MultiGraphSTModelGAT_PyG(nn.Module):
    """
    Spatiotemporal model with:
    1) Transformer for temporal dimension,
    2) Multi-graph adjacency (geo, corr, dynamic),
    3) GAT stack using PyTorch Geometric,
    4) Final projection => T_out day forecast.
    """
    def __init__(
        self,
        num_nodes,
        d_model=32,
        nhead=4,
        num_transformer_layers=2,
        hidden_dim_gnn=32,
        num_gnn_layers=2,
        dropout=0.1,
        device='cpu',
        gat_heads=1
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device

        # 1) Transformer
        self.transformer = TransformerTemporalEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_transformer_layers,
            dim_feedforward=4*d_model, dropout=dropout
        )

        # 2) Dynamic adjacency
        self.graph_learner = GraphLearner(d_model, alpha=1.0)

        # Gating parameters for multi-graph
        self.g_geo  = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.g_corr = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # GAT stack using PyTorch Geometric's GATConv
        self.num_gnn_layers = num_gnn_layers
        self.gnn_layers = nn.ModuleList([
            pyg_nn.GATConv(
                in_channels=d_model if i==0 else hidden_dim_gnn * gat_heads,
                out_channels=hidden_dim_gnn,
                heads=gat_heads,
                dropout=dropout,
                concat=True  # Concatenate heads' outputs
            )
            for i in range(num_gnn_layers)
        ])

        # Final projection for forecast
        self.output_fc = nn.Linear(hidden_dim_gnn * gat_heads, num_timesteps_output)

    def forward(self, X, A_geo, A_corr):
        """
        X => [B, T_in, m, C]
        A_geo => [m, m] on device
        A_corr=> [m, m] on device
        Output => [B, T_out, m]
        """
        B, T_in, m, C = X.shape

        # 1) Get node embeddings from transformer => [B, m, d_model]
        node_emb = self.transformer(X)

        # 2) Dynamic adjacency => [B, m, m]
        A_dyn = self.graph_learner(node_emb)

        # 3) Fuse multi-graph
        A_geo_b = A_geo.unsqueeze(0).repeat(B,1,1)   # [B, m, m]
        A_corr_b = A_corr.unsqueeze(0).repeat(B,1,1) # [B, m, m]
        A_fused = self.g_geo * A_geo_b + self.g_corr * A_corr_b + A_dyn
        A_fused = torch.clamp(A_fused, 0, 1)

        # 4) Convert fused adjacency matrices to edge_index and batch
        edge_index, batch = get_batched_edge_index(A_fused, B)  # [2, E_total], [B * m]

        # 5) Flatten node embeddings for all graphs in the batch
        x = node_emb.view(B * m, -1)  # [B * m, d_model]

        # 6) Apply GAT layers
        for idx, gat in enumerate(self.gnn_layers):
            x = gat(x, edge_index)  # [B * m, hidden_dim_gnn * heads]
            x = F.elu(x)            # Activation

        # 7) Reshape back to [B, m, hidden_dim_gnn * heads]
        x = x.view(B, m, -1)  # [B, m, hidden_dim_gnn * heads]

        # 8) Final projection => [B, m, T_out]
        out_m = self.output_fc(x)  # [B, m, T_out]
        out = out_m.permute(0,2,1) # [B, T_out, m]
        return out

# ==============================================================================
# 4. Training, Validation, Testing, and Plotting
# ==============================================================================
def initialize_model_gat_pyg(num_nodes, device, gat_heads=1):
    """
    Initializes the MultiGraphSTModelGAT_PyG model with specified hyperparameters.

    Args:
        num_nodes (int): Number of nodes (regions).
        device (torch.device): Device to run the model on.
        gat_heads (int): Number of attention heads in GAT layers.

    Returns:
        model, optimizer, criterion, scheduler
    """
    model = MultiGraphSTModelGAT_PyG(
        num_nodes=num_nodes,
        d_model=32,
        nhead=4,
        num_transformer_layers=2,
        hidden_dim_gnn=32,
        num_gnn_layers=2,
        dropout=0.1,
        device=device,
        gat_heads=gat_heads
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    return model, optimizer, criterion, scheduler

def train_val_test_pyg(
    model, optimizer, criterion, scheduler,
    train_loader, val_loader, test_loader,
    A_geo, A_corr,
    experiment_id=1
):
    """
    Train the model and evaluate on validation and test sets.

    Args:
        model (nn.Module): The spatiotemporal model.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        test_loader (DataLoader): Test data loader.
        A_geo (torch.Tensor): Geographic adjacency matrix.
        A_corr (torch.Tensor): Correlation adjacency matrix.
        experiment_id (int): Identifier for the experiment.

    Returns:
        dict: Dictionary containing test loss and various metrics.
    """
    best_val_loss = float('inf')
    patience_count = 0
    train_losses, val_losses = [], []
    summary_metrics = []

    for epoch in range(num_epochs):
        # --- TRAIN ---
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            optimizer.zero_grad()
            pred = model(batch_X, A_geo, A_corr)  # [B, T_out, m]
            loss = criterion(pred, batch_Y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- VAL ---
        model.eval()
        epoch_val_loss = 0.0
        all_val_preds = []
        all_val_actuals = []
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                val_pred = model(batch_X, A_geo, A_corr)
                vloss = criterion(val_pred, batch_Y)
                epoch_val_loss += vloss.item()

                all_val_preds.append(val_pred.cpu().numpy())
                all_val_actuals.append(batch_Y.cpu().numpy())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(f"models/experiment{experiment_id}", exist_ok=True)
            torch.save(model.state_dict(), f"models/experiment{experiment_id}/best_model.pth")
            patience_count = 0
            print("[Info] New best model saved.")
        else:
            patience_count += 1
            if patience_count >= early_stopping_patience:
                print("[Info] Early stopping triggered.")
                break

    # Plot Training vs Validation Loss
    plt.figure(figsize=(10,6))
    sns.lineplot(x=range(1,len(train_losses)+1), y=train_losses, marker='o', label='Train Loss')
    sns.lineplot(x=range(1,len(val_losses)+1), y=val_losses, marker='x', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f"Training vs Validation Loss - Experiment {experiment_id}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'figures/train_val_loss_experiment{experiment_id}.png', dpi=300)
    plt.show()

    # --- TEST ---
    model.load_state_dict(torch.load(f"models/experiment{experiment_id}/best_model.pth", map_location=device))
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_actuals= []
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            test_pred = model(batch_X, A_geo, A_corr)
            t_loss = criterion(test_pred, batch_Y)
            test_loss += t_loss.item()
            all_preds.append(test_pred.cpu())
            all_actuals.append(batch_Y.cpu())

    avg_test_loss = test_loss / len(test_loader)
    print(f"[Experiment {experiment_id}] Test MSE Loss: {avg_test_loss:.6f}")

    all_preds = torch.cat(all_preds, dim=0).numpy()    # [samples, T_out, m]
    all_actuals = torch.cat(all_actuals, dim=0).numpy()

    # Inverse transform if scaler was used
    if dataset.scaler is not None:
        # Assuming scaler was fitted on 'covidOccupiedMVBeds' only
        covid_idx = dataset.features.index('covidOccupiedMVBeds')
        all_preds_scaled = all_preds.copy()
        all_actuals_scaled = all_actuals.copy()

        # Inverse scaling only for 'covidOccupiedMVBeds'
        all_preds_scaled[:,:,covid_idx] = dataset.scaler.inverse_transform(
            all_preds_scaled[:,:,covid_idx].reshape(-1, 1)
        ).reshape(all_preds_scaled[:,:,covid_idx].shape)
        all_actuals_scaled[:,:,covid_idx] = dataset.scaler.inverse_transform(
            all_actuals_scaled[:,:,covid_idx].reshape(-1, 1)
        ).reshape(all_actuals_scaled[:,:,covid_idx].shape)
    else:
        all_preds_scaled = all_preds
        all_actuals_scaled = all_actuals

    # Compute Metrics for Each Forecast Horizon on Inverse Scaled Data
    metrics_dict = {
        'Forecast_Horizon': [],
        'Region': [],
        'MAE': [],
        'MSE': [],
        'RMSE': [],
        'Pearson_CC': []
    }

    for horizon in forecast_horizons:
        preds_h = all_preds_scaled[:, :horizon, :]          # [samples, horizon, m]
        actuals_h = all_actuals_scaled[:, :horizon, :]      # [samples, horizon, m]

        # Flatten for metrics
        preds_flat = preds_h.reshape(-1, preds_h.shape[2])    # [samples * horizon, m]
        actuals_flat = actuals_h.reshape(-1, actuals_h.shape[2])  # [samples * horizon, m]

        mae_node = mean_absolute_error(actuals_flat, preds_flat, multioutput='raw_values')
        mse_node = mean_squared_error(actuals_flat, preds_flat, multioutput='raw_values')
        rmse_node = np.sqrt(mse_node)
        pearson_node = []
        for i in range(actuals_flat.shape[1]):
            if np.std(preds_flat[:, i]) < 1e-9 or np.std(actuals_flat[:, i]) < 1e-9:
                pearson_node.append(0)
            else:
                cc, _ = pearsonr(preds_flat[:, i], actuals_flat[:, i])
                pearson_node.append(cc)

        for idx, region in enumerate(dataset.regions):
            metrics_dict['Forecast_Horizon'].append(horizon)
            metrics_dict['Region'].append(region)
            metrics_dict['MAE'].append(mae_node[idx])
            metrics_dict['MSE'].append(mse_node[idx])
            metrics_dict['RMSE'].append(rmse_node[idx])
            metrics_dict['Pearson_CC'].append(pearson_node[idx])

    # Save Metrics to CSV
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_csv_path = f'results/metrics/metrics_experiment{experiment_id}.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[Info] Metrics saved to {metrics_csv_path}")

    # Plot Overall Metrics for Each Forecast Horizon
    def plot_overall_metrics(metrics_df, experiment_id):
        """
        Plots average metrics across different forecast horizons.
        """
        metrics_to_plot = ['MAE', 'MSE', 'RMSE', 'Pearson_CC']
        plt.figure(figsize=(20, 15))
        for i, metric in enumerate(metrics_to_plot, 1):
            plt.subplot(2, 2, i)
            # Calculate average per Forecast_Horizon
            avg_metrics = metrics_df.groupby('Forecast_Horizon')[metric].mean().reset_index()
            sns.barplot(x='Forecast_Horizon', y=metric, data=avg_metrics)
            plt.title(f'Average {metric} across Forecast Horizons')
            plt.ylabel(metric)
            plt.xlabel('Forecast Horizon (days)')
            for idx, row in avg_metrics.iterrows():
                plt.text(row.name, row[metric] + 0.01 * row[metric], f"{row[metric]:.2f}",
                         ha='center', va='bottom', fontsize=12)
        plt.tight_layout()
        summary_plot = f'figures/summary_metrics_experiment{experiment_id}.png'
        plt.savefig(summary_plot, dpi=300)
        plt.show()
        print(f"[Info] Summary metrics plot saved to {summary_plot}")

    plot_overall_metrics(metrics_df, experiment_id)
    print(f"[Info] Final Results: {metrics_dict}")

    return {
        'test_loss': avg_test_loss,
        'MAE': metrics_dict['MAE'],
        'MSE': metrics_dict['MSE'],
        'RMSE': metrics_dict['RMSE'],
        'Pearson_CC': metrics_dict['Pearson_CC']
    }

# ==============================================================================
# 5. Main Execution
# ==============================================================================
if __name__ == "__main__":
    # A) Load your data
    csv_path = "data/merged_nhs_covid_data.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = pd.read_csv(csv_path, parse_dates=['date'])

    # Example reference coordinates:
    reference_coordinates = {
        "East of England": (52.1766, 0.425889),
        "Midlands": (52.7269, -1.458210),
        "London": (51.4923, -0.308660),
        "South East": (51.4341, -0.969570),
        "South West": (50.8112, -3.633430),
        "North West": (53.8981, -2.657550),
        "North East and Yorkshire": (54.5378, -2.180390),
    }

    # B) Build adjacency from geography & correlation
    data_processed, A_geo, A_corr = load_and_compute_adjacencies(
        data,
        reference_coordinates,
        correlation_feature='covidOccupiedMVBeds',
        corr_window=None  # or some integer like 180 for last 180 days
    )

    # IMPORTANT FIX: move adjacency to device
    A_geo = A_geo.to(device)
    A_corr = A_corr.to(device)

    # C) Build dataset => train/val/test
    # Initialize dataset without scaling
    initial_dataset = MyCovidDataset(
        data_processed,
        num_timesteps_input=num_timesteps_input,
        num_timesteps_output=num_timesteps_output,
        scaler=None
    )
    total_len = len(initial_dataset)
    train_size = int(0.7 * total_len)
    val_size   = int(0.15 * total_len)
    test_size  = total_len - train_size - val_size
    train_indices = list(range(0, train_size))
    val_indices   = list(range(train_size, train_size + val_size))
    test_indices  = list(range(train_size + val_size, total_len))

    # Create train subset to fit the scaler
    train_subset_initial = Subset(initial_dataset, train_indices)
    scaler = MinMaxScaler()
    # Extract training data for scaler (only 'covidOccupiedMVBeds')
    train_Y = []
    for i in range(len(train_subset_initial)):
        _, batch_Y = train_subset_initial[i]
        train_Y.append(batch_Y.numpy())
    train_Y = np.concatenate(train_Y, axis=0).reshape(-1, 1)  # [samples*T_out*m, 1]
    scaler.fit(train_Y)

    # Now, create the full dataset with scaling
    dataset = MyCovidDataset(
        data_processed,
        num_timesteps_input=num_timesteps_input,
        num_timesteps_output=num_timesteps_output,
        scaler=scaler
    )
    # Recreate subsets with scaled data
    train_subset = Subset(dataset, train_indices)
    val_subset   = Subset(dataset, val_indices)
    test_subset  = Subset(dataset, test_indices)

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_subset,  batch_size=batch_size, shuffle=False, drop_last=False)

    print(f"[Info] Train/Val/Test sizes: {len(train_subset)} {len(val_subset)} {len(test_subset)}")

    # D) Initialize model
    num_nodes = dataset.num_nodes
    model, optimizer, criterion, scheduler = initialize_model_gat_pyg(num_nodes, device, gat_heads=1)

    # E) Train/Val/Test
    results = train_val_test_pyg(
        model, optimizer, criterion, scheduler,
        train_loader, val_loader, test_loader,
        A_geo, A_corr,
        experiment_id=1
    )

    print(f"[Info] Final Results: {results}")
