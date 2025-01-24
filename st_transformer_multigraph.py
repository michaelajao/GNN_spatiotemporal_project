#!/usr/bin/env python
# coding: utf-8

"""
st_transformer_multigraph.py

Spatiotemporal Transformer + Multi-Graph GNN (Prototype)
========================================================

This script implements a novel framework that:
1) Replaces EpiGNN's temporal convolution with a Transformer encoder (for global temporal context).
2) Incorporates multiple adjacency matrices:
   - Geographic adjacency (A_geo)
   - Correlation adjacency (A_corr)
   - Dynamic adjacency (A_dyn) learned from node embeddings
   Then fuses them with a gating mechanism to create a final adjacency.

The code includes:
- Data preprocessing (7-day rolling means, pivoting, scaling)
- Correlation adjacency computation
- Transformer-based temporal encoder
- Multi-graph fusion (geo, corr, dynamic)
- A GNN to handle the final spatiotemporal embeddings
- Train/val/test loops with metrics: MAE, MSE, RMSE, R², Pearson CC
- Plotting of losses, time series, and adjacency

Author: [Your Name or Group], [Year]
License: [MIT or other]
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
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Parameter
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates

# Matplotlib style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'figure.figsize': (10, 6),
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'font.size': 12,
    'figure.dpi': 300
})

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[Info] Using device: {device}")

# Hyperparams
num_features = 5  # [new_confirmed, new_deceased, newAdmissions, hospitalCases, covidOccupiedMVBeds]
num_timesteps_input = 14
num_timesteps_output = 7
learning_rate = 1e-4
batch_size = 32
num_epochs = 300
early_stopping_patience = 20
threshold_distance = 300.0  # for geographic adjacency

# Directories
os.makedirs('figures', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

# ==============================================================================
# 1. Data Preprocessing & Adjacency
# ==============================================================================
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371
    return c*r

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

    # Region list
    regions = pivot_df.columns.tolist()
    m = len(regions)

    # 1) Geographic adjacency
    latitudes = []
    longitudes = []
    for reg in regions:
        # from reference coords
        lat, lon = reference_coordinates[reg]
        latitudes.append(lat)
        longitudes.append(lon)

    A_geo_np = np.zeros((m,m), dtype=np.float32)
    for i in range(m):
        for j in range(m):
            if i == j:
                A_geo_np[i,j] = 1.
            else:
                dist = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                if dist <= threshold_distance:
                    A_geo_np[i,j] = 1.
                    A_geo_np[j,i] = 1.

    A_geo = torch.tensor(A_geo_np, dtype=torch.float32)

    # 2) Correlation adjacency
    data_for_corr = pivot_df.values.T  # shape [m, T]
    corr_matrix = np.corrcoef(data_for_corr)  # shape [m, m]
    np.fill_diagonal(corr_matrix, 1.0)
    # Optionally clip negatives
    # corr_matrix = np.clip(corr_matrix, 0, 1)
    A_corr_np = corr_matrix.astype(np.float32)
    A_corr = torch.tensor(A_corr_np, dtype=torch.float32)

    return data, A_geo, A_corr

class MyCovidDataset(Dataset):
    """
    Similar to EpiGNN: 
      X => (T_in, m, F)
      Y => (T_out, m)
    We'll do a standard scaling at the dataset level if needed.
    """

    def __init__(self, data, num_timesteps_input=14, num_timesteps_output=7, scaler=None):
        super().__init__()
        self.data = data.copy()
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output

        # Region ordering
        self.regions = data['areaName'].unique()
        self.num_nodes = len(self.regions)
        self.region_to_idx = {r: i for i, r in enumerate(self.regions)}
        self.data['region_idx'] = self.data['areaName'].map(self.region_to_idx)

        # features
        self.features = ['new_confirmed','new_deceased','newAdmissions','hospitalCases','covidOccupiedMVBeds']

        # pivot => shape: [date, region_idx, feats]
        pivot_df = self.data.pivot(index='date', columns='region_idx', values=self.features)
        pivot_df = pivot_df.fillna(0.0)
        pivot_df.sort_index(inplace=True)

        self.num_features = len(self.features)
        self.num_dates = pivot_df.shape[0]
        self.feature_array = pivot_df.values.reshape(self.num_dates, self.num_nodes, self.num_features)

        if scaler is not None:
            self.scaler = scaler
            arr_2d = self.feature_array.reshape(-1, self.num_features)
            arr_2d = self.scaler.transform(arr_2d)
            self.feature_array = arr_2d.reshape(self.num_dates, self.num_nodes, self.num_features)
        else:
            self.scaler = None

    def __len__(self):
        return self.num_dates - self.num_timesteps_input - self.num_timesteps_output + 1

    def __getitem__(self, idx):
        X = self.feature_array[idx : idx + self.num_timesteps_input]  # (T_in, m, F)
        Y = self.feature_array[idx + self.num_timesteps_input : idx + self.num_timesteps_input + self.num_timesteps_output, :, 4]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# ==============================================================================
# 2. Model Definitions
# ==============================================================================
class TransformerTemporalEncoder(nn.Module):
    """
    A small Transformer encoder that processes each region's time series.

    Input shape to this module (for a single sample):
      X: [T_in, F]
    But we may handle [B*m, T_in, F] if we combine region with batch or do it region by region.

    Here we'll assume input shape => [B, T_in, m, F] and flatten (B*m, T_in, F)
    Then add position embedding, pass through a standard nn.TransformerEncoder.

    For simplicity, we do a single block or a few layers with multi-head attention.
    """
    def __init__(self, d_model=32, nhead=4, num_layers=2, dim_feedforward=64, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # to map input F -> d_model
        # (we might have F=5 in your case, so let's map 5 -> d_model=32)
        self.input_proj = nn.Linear(5, d_model)

    def forward(self, X):
        """
        X shape => [B, T_in, m, F]
        We'll flatten region into the batch dimension => new batch = B*m
        => shape => [B*m, T_in, F], then project F->d_model, and reorder to [T_in, B*m, d_model]
        => pass through transformer => => [T_in, B*m, d_model]
        => we might only want the final hidden or average pool.

        We'll return => [B, m, d_model] as the embedding.
        """
        B, T_in, m, F = X.shape
        X_reshaped = X.permute(0,2,1,3)  # => [B, m, T_in, F]
        # flatten B and m => Bm = B*m
        Bm = B*m
        X_reshaped = X_reshaped.reshape(Bm, T_in, F)  # [Bm, T_in, F]

        # project input feats => d_model
        X_proj = self.input_proj(X_reshaped)  # => [Bm, T_in, d_model]

        # For nn.TransformerEncoder, we want shape => [T_in, Bm, d_model]
        X_proj = X_proj.permute(1,0,2)  # => [T_in, Bm, d_model]

        # pass through the transformer
        out_tf = self.transformer_encoder(X_proj)  # => [T_in, Bm, d_model]

        # let's average pool over T_in dimension or take last
        # we'll average => shape => [Bm, d_model]
        out_pool = torch.mean(out_tf, dim=0)

        # reshape => [B, m, d_model]
        out_final = out_pool.reshape(B, m, self.d_model)
        return out_final

class GraphLearner(nn.Module):
    """
    Learns adjacency from node embeddings (like your EpiGNN dynamic adjacency).
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
        x1 = torch.tanh(self.alpha * self.lin1(emb))  # [B, m, hd]
        x2 = torch.tanh(self.alpha * self.lin2(emb))
        adj = torch.bmm(x1, x2.transpose(1,2)) - torch.bmm(x2, x1.transpose(1,2))
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))
        return adj

class GraphConvLayer(nn.Module):
    """
    Basic GCN-like layer.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.bias = Parameter(torch.Tensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        bound = 1.0 / math.sqrt(out_features)
        self.bias.data.uniform_(-bound, bound)
        self.act = nn.ELU()

    def forward(self, x, adj):
        """
        x => [B, m, in_features]
        adj => [B, m, m]
        output => [B, m, out_features]
        """
        support = torch.matmul(x, self.weight)  # [B, m, out_features]
        out = torch.bmm(adj, support)
        out = out + self.bias
        return self.act(out)

def getLaplaceMat(B, m, adj):
    """
    same idea as before => row-normalize adjacency
    """
    i_mat = torch.eye(m, device=adj.device).unsqueeze(0).expand(B, m, m)
    adj_bin = (adj > 0).float()
    deg = torch.sum(adj_bin, dim=2)
    deg_inv = 1.0 / (deg + 1e-12)
    deg_inv_mat = i_mat * deg_inv.unsqueeze(2)
    laplace_mat = torch.bmm(deg_inv_mat, adj_bin)
    return laplace_mat

class MultiGraphSTModel(nn.Module):
    """
    This model does:
    1) TransformerTemporalEncoder => [B, m, d_model]
    2) Combine adjacency: A_geo, A_corr, A_dyn
    3) GNN block => final projection for T_out steps
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
        device='cpu'
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.device = device

        # 1) Transformer for temporal dimension
        self.transformer = TransformerTemporalEncoder(
            d_model=d_model, nhead=nhead, num_layers=num_transformer_layers,
            dim_feedforward=4*d_model, dropout=dropout
        )

        # 2) Graph learner for dynamic adjacency
        self.graph_learner = GraphLearner(d_model, alpha=1.0)  # uses the transformer's node embeddings

        # gating params for multi-graph
        # we'll do: A_fused = clamp(A_geo*g_geo + A_corr*g_corr + A_dyn,0,1)
        # or a more advanced approach
        self.g_geo  = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.g_corr = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))

        # GNN stack
        self.num_gnn_layers = num_gnn_layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(in_features=d_model, out_features=hidden_dim_gnn)
            if i == 0 else
            GraphConvLayer(in_features=hidden_dim_gnn, out_features=hidden_dim_gnn)
            for i in range(num_gnn_layers)
        ])

        # final projection => 7-day forecast
        # after gnn, shape => [B, m, hidden_dim_gnn], let's do a linear map => T_out
        self.output_fc = nn.Linear(hidden_dim_gnn, num_timesteps_output)

    def forward(self, X, A_geo, A_corr):
        """
        X => [B, T_in, m, F]
        A_geo => [m, m]
        A_corr => [m, m]

        We'll produce dynamic adjacency => shape [B, m, m].
        Then fuse them.
        Then pass embeddings through GNN.
        Return => [B, T_out, m]
        """
        B, T_in, m, F = X.shape
        # 1) get node embeddings from transformer
        node_emb = self.transformer(X)  # => [B, m, d_model]

        # 2) dynamic adjacency => [B, m, m]
        A_dyn = self.graph_learner(node_emb)  # => [B, m, m]

        # 3) fuse multi-graph
        # A_geo, A_corr => shape [m, m], broadcast => [B, m, m]
        A_geo_b = A_geo.unsqueeze(0).to(self.device).repeat(B,1,1)
        A_corr_b= A_corr.unsqueeze(0).to(self.device).repeat(B,1,1)
        # gating approach
        A_fused = self.g_geo*A_geo_b + self.g_corr*A_corr_b + A_dyn
        A_fused = torch.clamp(A_fused, 0, 1)

        lap_mat = getLaplaceMat(B, m, A_fused)

        # 4) GNN forward
        x = node_emb  # [B, m, d_model]
        for gnn in self.gnn_layers:
            x = gnn(x, lap_mat)

        # 5) final => [B, m, T_out]
        out_m = self.output_fc(x)  # => [B, m, T_out]
        out = out_m.permute(0,2,1) # => [B, T_out, m]
        return out

# ==============================================================================
# 3. Train/Val/Test Routines & Main
# ==============================================================================
def initialize_model(num_nodes):
    model = MultiGraphSTModel(num_nodes=num_nodes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    return model, optimizer, criterion, scheduler

def train_val_test(
    model, optimizer, criterion, scheduler,
    train_loader, val_loader, test_loader,
    A_geo, A_corr,
    experiment_id=1
):
    best_val_loss = float('inf')
    patience_count = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        # --- TRAIN ---
        model.train()
        epoch_train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

            optimizer.zero_grad()
            pred = model(batch_X, A_geo, A_corr)
            loss = criterion(pred, batch_Y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss/len(train_loader)
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

        avg_val_loss = epoch_val_loss/len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        # optional R2
        all_val_preds = np.concatenate(all_val_preds, axis=0)  # shape [samples, T_out, m]
        all_val_actuals = np.concatenate(all_val_actuals, axis=0)
        preds_2d = all_val_preds.reshape(-1, all_val_preds.shape[2])
        actuals_2d = all_val_actuals.reshape(-1, all_val_preds.shape[2])
        r2_vals = r2_score(actuals_2d, preds_2d, multioutput='raw_values')

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val R² (per node): {r2_vals}")

        # early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(f"models/experiment{experiment_id}", exist_ok=True)
            torch.save(model.state_dict(), f"models/experiment{experiment_id}/best_model.pth")
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= early_stopping_patience:
                print("[Info] Early stopping triggered.")
                break

    # Plot train vs val
    plt.figure(figsize=(8,5))
    sns.lineplot(x=range(1,len(train_losses)+1), y=train_losses, marker='o', label='Train')
    sns.lineplot(x=range(1,len(val_losses)+1),   y=val_losses,   marker='x', label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f"Train vs Val Loss - Experiment {experiment_id}")
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
    all_actuals = []
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            pred_test = model(batch_X, A_geo, A_corr)
            loss_test = criterion(pred_test, batch_Y)
            test_loss += loss_test.item()
            all_preds.append(pred_test.cpu())
            all_actuals.append(batch_Y.cpu())

    avg_test_loss = test_loss / len(test_loader)
    print(f"[Experiment {experiment_id}] Test MSE Loss: {avg_test_loss:.4f}")

    all_preds = torch.cat(all_preds, dim=0).numpy()   # shape [samples, T_out, m]
    all_actuals = torch.cat(all_actuals, dim=0).numpy()

    # compute metrics
    # flatten => [samples*T_out, m]
    preds_flat = all_preds.reshape(-1, all_preds.shape[2])
    actuals_flat= all_actuals.reshape(-1, all_preds.shape[2])

    mae_per_node = mean_absolute_error(actuals_flat, preds_flat, multioutput='raw_values')
    mse_per_node = mean_squared_error(actuals_flat, preds_flat, multioutput='raw_values')
    rmse_per_node= np.sqrt(mse_per_node)
    r2_per_node  = r2_score(actuals_flat, preds_flat, multioutput='raw_values')
    pearson_per_node = []
    for i in range(preds_flat.shape[1]):
        if np.std(preds_flat[:,i])<1e-9 or np.std(actuals_flat[:,i])<1e-9:
            pearson_per_node.append(0.0)
        else:
            cc,_ = pearsonr(preds_flat[:,i], actuals_flat[:,i])
            pearson_per_node.append(cc)

    # Print summary
    print("[Test] Node-level MAE:", mae_per_node)
    print("[Test] Node-level RMSE:", rmse_per_node)
    print("[Test] Node-level R2:", r2_per_node)
    print("[Test] Node-level Pearson:", pearson_per_node)

    # return for further usage
    return {
        'test_loss': avg_test_loss,
        'mae': mae_per_node,
        'mse': mse_per_node,
        'rmse': rmse_per_node,
        'r2': r2_per_node,
        'pearson': pearson_per_node
    }

# ==============================================================================
# 4. Main Example
# ==============================================================================
if __name__ == "__main__":
    # A) Load your data
    csv_path = "data/merged_nhs_covid_data.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    data = pd.read_csv(csv_path, parse_dates=['date'])

    # Example reference coords:
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
        corr_window=None  # or some int
    )

    # C) Build dataset => train/val/test
    initial_dataset = MyCovidDataset(
        data_processed,
        num_timesteps_input=num_timesteps_input,
        num_timesteps_output=num_timesteps_output,
        scaler=None  # we'll do standard scaling if needed
    )
    total_len = len(initial_dataset)
    train_size = int(0.7*total_len)
    val_size   = int(0.15*total_len)
    test_size  = total_len - train_size - val_size
    train_indices = list(range(0, train_size))
    val_indices   = list(range(train_size, train_size+val_size))
    test_indices  = list(range(train_size+val_size, total_len))

    train_subset = Subset(initial_dataset, train_indices)
    val_subset   = Subset(initial_dataset, val_indices)
    test_subset  = Subset(initial_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_subset,  batch_size=batch_size, shuffle=False, drop_last=False)

    print("[Info] Train/Val/Test sizes:", len(train_subset), len(val_subset), len(test_subset))

    # D) Initialize model
    # we get the number of nodes from the dataset
    num_nodes = initial_dataset.num_nodes
    model, optimizer, criterion, scheduler = initialize_model(num_nodes)

    # E) Train/Val/Test
    results = train_val_test(
        model, optimizer, criterion, scheduler,
        train_loader, val_loader, test_loader,
        A_geo, A_corr,
        experiment_id=1
    )

    print("[Info] Final Results:", results)
