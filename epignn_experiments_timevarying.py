#!/usr/bin/env python
# coding: utf-8

"""
epignn_experiments_timevarying.py

Extended EpiGNN Experiments: Static, Dynamic, Hybrid, and Time-Varying Adjacency Matrices
========================================================================================

This script conducts experiments to compare the performance of the EpiGNN model using
static, dynamic, hybrid, and time-varying adjacency matrices for forecasting COVID Occupied MV Beds.

Each experiment involves:
1. Training the model with the specified adjacency type.
2. Evaluating the model on the test set.
3. Saving metrics, models, and visualizations in dedicated folders.

Metrics computed and saved:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score
- Pearson Correlation Coefficient (PCC)

Plots generated:
- Training and Validation Loss Curves
- Actual vs. Predicted Time Series (per region and overall)
- Boxplots of Prediction Errors
- Scatter Plots of Actual vs. Predicted
- Summary bar charts comparing overall metrics (MAE, MSE, RMSE, R², PCC) across experiments
- Learned Adjacency Matrix Visualization

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
from torch.nn import Parameter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
k = 8             # Convolution channels
hidA = 32         # Dimension for Q/K transformations
hidR = 40         # Dimension for the GNN blocks
hidP = 1
n_layer = 6       # Number of GraphConv layers
dropout = 0.5
learning_rate = 1e-4
num_epochs = 500
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
    Creates a binary adjacency matrix based on distance threshold using Haversine formula.
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
                    adj_matrix[j][i] = 1  # Ensure symmetry

    return torch.tensor(adj_matrix, dtype=torch.float32)

def getLaplaceMat(batch_size: int,
                  m: int,
                  adj: torch.Tensor) -> torch.Tensor:
    """
    Compute a Laplacian-like matrix for GCN from adjacency.
    """
    i_mat = torch.eye(m).to(adj.device).unsqueeze(0).expand(batch_size, m, m)
    adj_bin = (adj > 0).float()
    deg = torch.sum(adj_bin, dim=2)
    deg_inv = 1.0 / (deg + 1e-12)
    deg_inv_mat = i_mat * deg_inv.unsqueeze(2)
    laplace_mat = torch.bmm(deg_inv_mat, adj_bin)
    return laplace_mat

# ==============================================================================
# 5. Model Definitions
# ==============================================================================
class GraphConvLayer(nn.Module):
    """
    Basic Graph Convolutional Layer with ELU activation.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvLayer, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.act = nn.ELU()
        nn.init.xavier_uniform_(self.weight)

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            stdv = 1.0 / math.sqrt(out_features)
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, feature, adj):
        # feature: [B, m, in_features]
        support = torch.matmul(feature, self.weight)   # => [B, m, out_features]
        output = torch.bmm(adj, support)               # => [B, m, out_features]
        if self.bias is not None:
            return self.act(output + self.bias)
        return self.act(output)

class GraphLearner(nn.Module):
    """
    Learns adjacency matrix via attention-like node embeddings.
    """
    def __init__(self, hidden_dim, tanhalpha=1):
        super(GraphLearner, self).__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

    def forward(self, embedding):
        # embedding: (B, m, hidR)
        nodevec1 = torch.tanh(self.alpha * self.linear1(embedding))  # [B, m, hidR]
        nodevec2 = torch.tanh(self.alpha * self.linear2(embedding))  # [B, m, hidR]
        # Compute adjacency as difference of outer products
        adj = (torch.bmm(nodevec1, nodevec2.transpose(1, 2)) -
               torch.bmm(nodevec2, nodevec1.transpose(1, 2)))  # [B, m, m]
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))  # [B, m, m]
        return adj

class ConvBranch(nn.Module):
    """
    Single branch for RegionAwareConv (Conv2D + optional pooling).
    """
    def __init__(self,
                 m: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 dilation_factor: int = 2,
                 hidP: int = 1,
                 isPool: bool = True):
        super(ConvBranch, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=(kernel_size, 1),
                              dilation=(dilation_factor, 1))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.isPool = isPool
        if self.isPool and hidP is not None:
            self.pooling = nn.AdaptiveMaxPool2d((hidP, m))
        self.activate = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.isPool and hasattr(self, 'pooling'):
            x = self.pooling(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(-1))
        return self.activate(x)

class RegionAwareConv(nn.Module):
    """
    Combines local, period, and global convolution branches for spatiotemporal features.
    """
    def __init__(self, nfeat, P, m, k, hidP, dilation_factor=2):
        super(RegionAwareConv, self).__init__()
        # Local Convs
        self.conv_l1 = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                  kernel_size=3, dilation_factor=1, hidP=hidP)
        self.conv_l2 = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                  kernel_size=5, dilation_factor=1, hidP=hidP)
        # Period Convs
        self.conv_p1 = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                  kernel_size=3, dilation_factor=dilation_factor, hidP=hidP)
        self.conv_p2 = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                  kernel_size=5, dilation_factor=dilation_factor, hidP=hidP)
        # Global Conv
        self.conv_g = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                 kernel_size=P, dilation_factor=1, hidP=None, isPool=False)
        self.activate = nn.Tanh()

    def forward(self, x):
        # x shape: [B, F, T_in, m]
        x_l1 = self.conv_l1(x)  # [B, k, m]
        x_l2 = self.conv_l2(x)  # [B, k, m]
        x_local = torch.cat([x_l1, x_l2], dim=1)  # [B, 2k, m]

        x_p1 = self.conv_p1(x)  # [B, k, m]
        x_p2 = self.conv_p2(x)  # [B, k, m]
        x_period = torch.cat([x_p1, x_p2], dim=1)  # [B, 2k, m]

        x_global = self.conv_g(x)  # [B, k, m]

        x_cat = torch.cat([x_local, x_period, x_global], dim=1)  # [B, 5k, m]
        return self.activate(x_cat).permute(0, 2, 1)  # [B, m, 5k]

class EpiGNN(nn.Module):
    """
    EpiGNN for spatiotemporal forecasting with adjacency types:
      - static, dynamic, hybrid, or time_varying
    """
    def __init__(self,
                 num_nodes,
                 num_features,
                 num_timesteps_input,
                 num_timesteps_output,
                 k=8,
                 hidA=32,
                 hidR=40,
                 hidP=1,
                 n_layer=1,
                 dropout=0.5,
                 device='cpu'):
        super(EpiGNN, self).__init__()
        self.device = device
        self.m = num_nodes
        self.w = num_timesteps_input
        self.hidR = hidR
        self.hidA = hidA
        self.hidP = hidP
        self.k = k
        self.n = n_layer
        self.dropout_layer = nn.Dropout(dropout)

        # RegionAwareConv backbone
        self.backbone = RegionAwareConv(nfeat=num_features, P=self.w,
                                        m=self.m, k=self.k,
                                        hidP=self.hidP)

        # Map backbone output (5k) to hidR
        self.backbone_linear = nn.Linear(5 * k, hidR)  # [B, m, 5k] -> [B, m, hidR=40]

        # Additional backbone for time-varying adjacency
        self.backbone_timestep = nn.Sequential(
            nn.Linear(num_features, hidR),
            nn.ReLU(),
            nn.Linear(hidR, hidR)
        )

        # Q/K transformations
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.t_enc = nn.Linear(1, self.hidR)
        self.s_enc = nn.Linear(1, self.hidR)

        # Gating parameter for hybrid adjacency
        self.d_gate = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)
        nn.init.xavier_uniform_(self.d_gate)

        # Graph learner for dynamic adjacency
        self.graphGen = GraphLearner(self.hidR)

        # GNN blocks
        self.GNNBlocks = nn.ModuleList([
            GraphConvLayer(in_features=self.hidR, out_features=self.hidR)
            for _ in range(self.n)
        ])

        # Final projection
        self.output = nn.Linear(self.hidR * self.n + self.hidR, num_timesteps_output)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                stdv = 1.0 / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, X, adj, adjacency_type='hybrid'):
        """
        Forward pass of the EpiGNN model.

        Parameters:
        - X: [B, T_in, m, F] input features
        - adj: [B, m, m] adjacency matrix (static or zeros)
        - adjacency_type: 'static', 'dynamic', 'hybrid', or 'time_varying'

        Returns:
        - out: [B, T_out, m] predicted COVID Occupied MV Beds
        """
        B = X.size(0)
        # 1) RegionAwareConv backbone processing
        X_reshaped = X.permute(0, 3, 1, 2)  # [B, F, T_in, m]
        temp_emb = self.backbone(X_reshaped)   # => [B, m, 5k]
        temp_emb = self.backbone_linear(temp_emb)  # => [B, m, hidR=40]

        # 2) Time-Varying Adjacency
        if adjacency_type == 'time_varying':
            # Learn adjacency for each time step and average them
            timewise_adjs = []
            for t in range(self.w):
                xt = X[:, t, :, :]  # [B, m, F]
                xt_emb = self.backbone_timestep(xt.reshape(-1, X.size(-1)))  # [B*m, hidR=40]
                xt_emb = xt_emb.reshape(B, self.m, self.hidR)  # [B, m, 40]
                adj_t = self.graphGen(xt_emb)  # [B, m, m]
                timewise_adjs.append(adj_t)
            learned_adj = torch.mean(torch.stack(timewise_adjs, dim=1), dim=1)  # [B, m, m]

        else:
            # Q/K transforms
            query = self.dropout_layer(self.WQ(temp_emb))  # [B, m, hidA=32]
            key   = self.dropout_layer(self.WK(temp_emb))  # [B, m, hidA=32]
            attn  = torch.bmm(query, key.transpose(1, 2))  # => [B, m, m]
            attn  = F.normalize(attn, dim=-1, p=2, eps=1e-12)
            attn  = torch.sum(attn, dim=-1, keepdim=True)  # => [B, m, 1]

            # Local transmission risk
            d = torch.sum(adj, dim=2, keepdim=True)  # => [B, m, 1]
            s_enc = self.dropout_layer(self.s_enc(d))  # => [B, m, hidR=40]

            t_enc = self.dropout_layer(self.t_enc(attn))  # => [B, m, hidR=40]
            feat_emb = temp_emb + t_enc + s_enc           # => [B, m, 40]

            # Learned adjacency
            learned_adj = self.graphGen(feat_emb)         # [B, m, m]

        # 3) Combine adjacency matrices based on adjacency_type
        if adjacency_type == 'static':
            combined_adj = adj  # [B, m, m]
        elif adjacency_type == 'dynamic':
            combined_adj = learned_adj                      # [B, m, m]
        elif adjacency_type == 'hybrid':
            # Gate how much to incorporate static adjacency
            d_mat = torch.sum(adj, dim=0, keepdim=True) * torch.sum(adj, dim=1, keepdim=True)  # [1, m, m]
            d_mat = torch.sigmoid(self.d_gate * d_mat).to(adj.device)  # [m, m]
            spatial_adj = d_mat * adj                            # [B, m, m] via broadcasting
            combined_adj = torch.clamp(learned_adj + spatial_adj, 0, 1)  # [B, m, m]
        elif adjacency_type == 'time_varying':
            combined_adj = learned_adj                          # [B, m, m]
        else:
            raise ValueError("Invalid adjacency_type specified.")

        # 4) Normalize adjacency
        laplace_adj = getLaplaceMat(B, self.m, combined_adj)  # [B, m, m]

        # 5) GNN Layers
        node_state = temp_emb                                # [B, m, 40]
        node_state_list = []
        for layer in self.GNNBlocks:
            node_state = self.dropout_layer(layer(node_state, laplace_adj))  # [B, m, 40]
            node_state_list.append(node_state)

        node_state_all = torch.cat(node_state_list, dim=-1)   # [B, m, 40*n_layer]
        node_state_all = torch.cat([node_state_all, temp_emb], dim=-1)  # [B, m, 40*n_layer + 40]

        # 6) Final Output
        out = self.output(node_state_all)                    # [B, m, T_out]
        out = out.transpose(1, 2)                            # [B, T_out, m]
        return out

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
print("[Info] Data loaded and preprocessed.")

initial_dataset = NHSRegionDataset(data,
                                   num_timesteps_input=num_timesteps_input,
                                   num_timesteps_output=num_timesteps_output,
                                   scaler=None)
print(f"[Info] Total samples in initial dataset: {len(initial_dataset)}")

total_len = len(initial_dataset)
train_size = int(0.7 * total_len)
val_size   = int(0.15 * total_len)
test_size  = total_len - train_size - val_size

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
print("[Info] Scaler fitted on training data.")

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
                          shuffle=True, drop_last=True)
val_loader   = DataLoader(val_subset,   batch_size=batch_size,
                          shuffle=False, drop_last=False)
test_loader  = DataLoader(test_subset,  batch_size=batch_size,
                          shuffle=False, drop_last=False)

# Adjacency matrix
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
plt.show()  # Changed from plt.close() to plt.show()
print("[Info] Geographic adjacency graph saved to 'figures/geographic_adjacency_graph_static.png'.")

# ==============================================================================
# 7. Initialize Model
# ==============================================================================
def initialize_model():
    model = EpiGNN(
        num_nodes=num_nodes,
        num_features=num_features,
        num_timesteps_input=num_timesteps_input,
        num_timesteps_output=num_timesteps_output,
        k=k,
        hidA=hidA,
        hidR=hidR,
        hidP=hidP,
        n_layer=n_layer,
        dropout=dropout,
        device=device
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.MSELoss()
    # Removed verbose=True to fix the deprecated warning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=3)
    return model, optimizer, criterion, scheduler

# ==============================================================================
# 8. Training & Evaluation Function
# ==============================================================================
def run_experiment(adjacency_type='hybrid', experiment_id=1, summary_metrics=[]):
    """
    Run a full training/validation/testing cycle for a given adjacency type:
    'static', 'dynamic', 'hybrid', or 'time_varying'.
    """
    print(f"\n[Experiment {experiment_id}] Starting with {adjacency_type.capitalize()} Adjacency...\n")
    model, optimizer, criterion, scheduler = initialize_model()

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()

            if adjacency_type == 'static':
                adj_input = adj_static.unsqueeze(0).repeat(batch_X.size(0), 1, 1)  # [B, m, m]
            elif adjacency_type == 'dynamic':
                adj_input = torch.zeros_like(adj_static).unsqueeze(0).repeat(batch_X.size(0), 1, 1)  # [B, m, m]
            elif adjacency_type == 'hybrid':
                adj_input = adj_static.unsqueeze(0).repeat(batch_X.size(0), 1, 1)  # [B, m, m]
            elif adjacency_type == 'time_varying':
                adj_input = adj_static.unsqueeze(0).repeat(batch_X.size(0), 1, 1)  # [B, m, m] Placeholder
            else:
                raise ValueError("Invalid adjacency_type. Choose 'static', 'dynamic', 'hybrid', or 'time_varying'.")

            pred = model(batch_X, adj_input, adjacency_type=adjacency_type)
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
                if adjacency_type == 'static':
                    adj_input = adj_static.unsqueeze(0).repeat(batch_X.size(0), 1, 1)
                elif adjacency_type == 'dynamic':
                    adj_input = torch.zeros_like(adj_static).unsqueeze(0).repeat(batch_X.size(0), 1, 1)
                elif adjacency_type == 'hybrid':
                    adj_input = adj_static.unsqueeze(0).repeat(batch_X.size(0), 1, 1)
                elif adjacency_type == 'time_varying':
                    adj_input = adj_static.unsqueeze(0).repeat(batch_X.size(0), 1, 1)
                else:
                    raise ValueError("Invalid adjacency_type. Choose 'static', 'dynamic', 'hybrid', or 'time_varying'.")

                pred = model(batch_X, adj_input, adjacency_type=adjacency_type)
                vloss = criterion(pred, batch_Y)
                epoch_val_loss += vloss.item()

                all_val_preds.append(pred.cpu().numpy())
                all_val_actuals.append(batch_Y.cpu().numpy())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        scheduler.step(avg_val_loss)

        # R² Score per node
        all_val_preds = np.concatenate(all_val_preds, axis=0)
        all_val_actuals = np.concatenate(all_val_actuals, axis=0)
        preds_2d = all_val_preds.reshape(-1, num_nodes)
        actuals_2d = all_val_actuals.reshape(-1, num_nodes)
        r2_vals = r2_score(actuals_2d, preds_2d, multioutput='raw_values')

        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val R² (per node): {r2_vals}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_dir = f'models/experiment{experiment_id}'
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = f'{checkpoint_dir}/best_model_{adjacency_type}.pth'
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
    plt.title(f'Training vs. Validation Loss - {adjacency_type.capitalize()} Adjacency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = f'figures/training_validation_loss_experiment{experiment_id}_{adjacency_type}.png'
    plt.savefig(plot_path, dpi=300)
    plt.show()  # Changed from plt.close() to plt.show()
    print(f"[Info] Loss curves saved to {plot_path}")

    # Load best model for testing
    model.load_state_dict(torch.load(f'models/experiment{experiment_id}/best_model_{adjacency_type}.pth', map_location=device))
    model.eval()

    # Testing
    test_loss = 0.0
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            if adjacency_type == 'static':
                adj_input = adj_static.unsqueeze(0).repeat(batch_X.size(0), 1, 1)
            elif adjacency_type == 'dynamic':
                adj_input = torch.zeros_like(adj_static).unsqueeze(0).repeat(batch_X.size(0), 1, 1)
            elif adjacency_type == 'hybrid':
                adj_input = adj_static.unsqueeze(0).repeat(batch_X.size(0), 1, 1)
            elif adjacency_type == 'time_varying':
                adj_input = adj_static.unsqueeze(0).repeat(batch_X.size(0), 1, 1)
            else:
                raise ValueError("Invalid adjacency_type. Choose 'static', 'dynamic', 'hybrid', or 'time_varying'.")

            pred = model(batch_X, adj_input, adjacency_type=adjacency_type)
            loss = criterion(pred, batch_Y)
            test_loss += loss.item()

            all_preds.append(pred.cpu())
            all_actuals.append(batch_Y.cpu())

    avg_test_loss = test_loss / len(test_loader)
    print(f"[Experiment {experiment_id} - {adjacency_type.capitalize()}] Test MSE Loss: {avg_test_loss:.4f}")

    # Reshape predictions and actuals
    all_preds = torch.cat(all_preds, dim=0).numpy()   # [num_test_samples, T_out, m]
    all_actuals = torch.cat(all_actuals, dim=0).numpy()  # [num_test_samples, T_out, m]

    # Inverse transform if scaler was used
    if scaled_dataset.scaler is not None:
        scale_covid = scaled_dataset.scaler.scale_[4]  # 'covidOccupiedMVBeds' index
        mean_covid  = scaled_dataset.scaler.mean_[4]
        all_preds_np   = all_preds * scale_covid + mean_covid
        all_actuals_np = all_actuals * scale_covid + mean_covid
    else:
        all_preds_np   = all_preds
        all_actuals_np = all_actuals

    # Flatten for metric calculations
    preds_flat   = all_preds_np.reshape(-1, num_nodes)
    actuals_flat = all_actuals_np.reshape(-1, num_nodes)

    # Compute Metrics
    mae_per_node = mean_absolute_error(actuals_flat, preds_flat, multioutput='raw_values')
    mse_per_node = mean_squared_error(actuals_flat, preds_flat, multioutput='raw_values')
    rmse_per_node = np.sqrt(mse_per_node)
    r2_per_node  = r2_score(actuals_flat, preds_flat, multioutput='raw_values')

    pearson_per_node = []
    for i in range(num_nodes):
        if np.std(preds_flat[:, i]) < 1e-9 or np.std(actuals_flat[:, i]) < 1e-9:
            pearson_cc = 0
        else:
            pearson_cc, _ = pearsonr(preds_flat[:, i], actuals_flat[:, i])
        pearson_per_node.append(pearson_cc)

    # Organize metrics into a dictionary
    metrics_dict = {
        'Experiment_ID': [],
        'Adjacency_Type': [],
        'Region': [],
        'MAE': [],
        'MSE': [],
        'RMSE': [],
        'R2_Score': [],
        'Pearson_CC': []
    }

    for idx, region in enumerate(regions):
        metrics_dict['Experiment_ID'].append(experiment_id)
        metrics_dict['Adjacency_Type'].append(adjacency_type)
        metrics_dict['Region'].append(region)
        metrics_dict['MAE'].append(mae_per_node[idx])
        metrics_dict['MSE'].append(mse_per_node[idx])
        metrics_dict['RMSE'].append(rmse_per_node[idx])
        metrics_dict['R2_Score'].append(r2_per_node[idx])
        metrics_dict['Pearson_CC'].append(pearson_per_node[idx])

        # Accumulate summary metrics
        summary_metrics.append({
            'Experiment_ID': experiment_id,
            'Adjacency_Type': adjacency_type,
            'Region': region,
            'MAE': mae_per_node[idx],
            'MSE': mse_per_node[idx],
            'RMSE': rmse_per_node[idx],
            'R2_Score': r2_per_node[idx],
            'Pearson_CC': pearson_per_node[idx]
        })

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_csv_path = f'results/metrics/metrics_experiment{experiment_id}_{adjacency_type}.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[Info] Metrics saved to {metrics_csv_path}")

    # Visualization Steps
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
    def plot_time_series(region, df, adjacency_type, experiment_id):
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='Date', y=f'{region}_Actual', data=df,
                     label='Actual', color='blue', marker='o')
        sns.lineplot(x='Date', y=f'{region}_Predicted', data=df,
                     label='Predicted', color='red', linestyle='--', marker='x')
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.title(f'Actual vs Predicted COVID Occupied MV Beds - {region} ({adjacency_type.capitalize()})')
        plt.xlabel('Date')
        plt.ylabel('COVID Occupied MV Beds')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = f'figures/actual_vs_predicted_experiment{experiment_id}_{adjacency_type}_{region.replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()  # Changed from plt.close() to plt.show()
        print(f"[Info] Time series plot saved to {plot_path}")

    for region in regions:
        plot_time_series(region, merged_df, adjacency_type, experiment_id)

    # Overall Metrics Plot
    def plot_overall_metrics(summary_metrics, experiment_id, adjacency_type):
        """
        Aggregates and plots overall metrics (MAE, MSE, RMSE, R2, PCC) across all regions.
        """
        df = pd.DataFrame(summary_metrics)
        df = df[(df['Experiment_ID'] == experiment_id) & (df['Adjacency_Type'] == adjacency_type)]
        aggregated_metrics = df[['MAE','MSE','RMSE','R2_Score','Pearson_CC']].mean()

        plt.figure(figsize=(10, 6))
        sns.barplot(x=aggregated_metrics.index, y=aggregated_metrics.values, palette='Set2')
        plt.title(f'Overall Metrics (Averaged) - Experiment {experiment_id} ({adjacency_type.capitalize()})')
        plt.ylabel('Metric Value')
        plt.xlabel('Metric')
        for i, v in enumerate(aggregated_metrics.values):
            plt.text(i, v + 0.01 * max(aggregated_metrics.values), f"{v:.4f}",
                     ha='center', va='bottom', fontsize=12)
        plt.tight_layout()
        plot_path = f'figures/overall_metrics_experiment{experiment_id}_{adjacency_type}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()  # Changed from plt.close() to plt.show()
        print(f"[Info] Overall metrics bar chart saved to {plot_path}")

    plot_overall_metrics(summary_metrics, experiment_id, adjacency_type)

    # Boxplot of Errors
    def plot_error_boxplot(df, adjacency_type, experiment_id):
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
        plt.title(f'Prediction Errors Boxplot ({adjacency_type.capitalize()} Adjacency)')
        plt.xlabel('Region')
        plt.ylabel('Prediction Error (Pred - Actual)')
        plt.tight_layout()
        plot_path = f'figures/boxplot_errors_experiment{experiment_id}_{adjacency_type}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()  # Changed from plt.close() to plt.show()
        print(f"[Info] Boxplot of errors saved to {plot_path}")

    plot_error_boxplot(merged_df, adjacency_type, experiment_id)

    # Scatter Actual vs Predicted (Overall)
    def plot_scatter_actual_vs_predicted_overall(df, adjacency_type, experiment_id):
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
        plt.title(f'Overall Actual vs Predicted (Avg) ({adjacency_type.capitalize()})')
        plt.xlabel('Actual COVID Occupied MV Beds')
        plt.ylabel('Predicted COVID Occupied MV Beds')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = f'figures/scatter_actual_vs_predicted_overall_experiment{experiment_id}_{adjacency_type}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()  # Changed from plt.close() to plt.show()
        print(f"[Info] Overall scatter plot saved to {plot_path}")

    plot_scatter_actual_vs_predicted_overall(merged_df, adjacency_type, experiment_id)

    # Save final model
    final_model_path = f'models/experiment{experiment_id}/epignn_final_model_{adjacency_type}.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"[Info] Final model saved as '{final_model_path}'.")

    # Learned Adjacency Visualization
    def plot_learned_adjacency(model, adj_static, regions, adjacency_type, experiment_id):
        """
        Show the learned adjacency for a single test sample.
        """
        example_X, _ = test_loader.dataset[0]
        example_X = example_X.unsqueeze(0).to(device)  # [1, T_in, m, F]

        with torch.no_grad():
            # Pass through the model to get learned adjacency
            if adjacency_type == 'time_varying':
                learned_adj = []
                for t in range(num_timesteps_input):
                    xt = example_X[:, t, :, :]  # [1, m, F]
                    xt_emb = model.backbone_timestep(xt.reshape(-1, xt.size(-1)))  # [B*m, hidR=40]
                    xt_emb = xt_emb.reshape(1, num_nodes, hidR)  # [1, m, hidR=40]
                    adj_t = model.graphGen(xt_emb)  # [1, m, m]
                    learned_adj.append(adj_t)
                learned_adj = torch.mean(torch.stack(learned_adj, dim=1), dim=1).cpu().numpy()[0]  # [m, m]
                combined_adj = learned_adj
            else:
                # For static/dynamic/hybrid
                _ = model(example_X, adj_static.unsqueeze(0), adjacency_type=adjacency_type)
                if adjacency_type in ['dynamic','hybrid']:
                    # get learned adjacency
                    temp_emb = model.backbone(example_X.permute(0,3,1,2))
                    temp_emb = model.backbone_linear(temp_emb)
                    feat_emb = temp_emb  # Assuming no additional encodings
                    learned_adj = model.graphGen(feat_emb).cpu().numpy()[0]
                    if adjacency_type == 'hybrid':
                        combined_adj = learned_adj + adj_static.cpu().numpy()
                    else:
                        combined_adj = learned_adj
                else:
                    combined_adj = adj_static.cpu().numpy()

            # Plot heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(combined_adj, annot=True, fmt=".2f", cmap='viridis',
                        xticklabels=regions, yticklabels=regions)
            plt.title(f'Learned Adjacency Matrix ({adjacency_type.capitalize()}) - Experiment {experiment_id}')
            plt.xlabel('Regions')
            plt.ylabel('Regions')
            plt.tight_layout()
            plot_path = f'figures/learned_adjacency_matrix_experiment{experiment_id}_{adjacency_type}.png'
            plt.savefig(plot_path, dpi=300)
            plt.show()  # Changed from plt.close() to plt.show()
            print(f"[Info] Learned adjacency plot saved to {plot_path}")

    plot_learned_adjacency(model, adj_static, regions, adjacency_type, experiment_id)

    print(f"[Experiment {experiment_id}] Workflow complete.\n")

# ==============================================================================
# 9. Main: Run All Experiments
# ==============================================================================
if __name__ == "__main__":
    experiments = [
        {'adjacency_type': 'static',       'experiment_id': 1},
        {'adjacency_type': 'dynamic',      'experiment_id': 2},
        {'adjacency_type': 'hybrid',       'experiment_id': 3},
        {'adjacency_type': 'time_varying', 'experiment_id': 4},
    ]

    summary_metrics = []

    # Execute each experiment
    for exp in experiments:
        run_experiment(adjacency_type=exp['adjacency_type'],
                       experiment_id=exp['experiment_id'],
                       summary_metrics=summary_metrics)

    # Summarize results
    summary_df = pd.DataFrame(summary_metrics)
    summary_csv = 'results/metrics/summary_metrics_all.csv'
    summary_df.to_csv(summary_csv, index=False)
    print(f"[Info] Summary metrics saved to {summary_csv}")

    # Pivot to show average metrics per experiment
    summary_pivot = summary_df.groupby(['Experiment_ID', 'Adjacency_Type']).agg({
        'MAE': 'mean',
        'MSE': 'mean',
        'RMSE': 'mean',
        'R2_Score': 'mean',
        'Pearson_CC': 'mean'
    }).reset_index()

    print("\nSummary of All Experiments:\n", summary_pivot)

    summary_pivot_path = 'results/metrics/summary_metrics_pivot_all.csv'
    summary_pivot.to_csv(summary_pivot_path, index=False)
    print(f"[Info] Summary pivot table saved to {summary_pivot_path}")

    # Optional comparison plot for summary metrics
    def plot_summary_metrics(summary_pivot):
        """
        Plots average metrics across different adjacency types.
        """
        metrics_list = ['MAE', 'MSE', 'RMSE', 'R2_Score', 'Pearson_CC']
        plt.figure(figsize=(18, 12))
        for i, metric in enumerate(metrics_list, 1):
            plt.subplot(2, 3, i)
            sns.barplot(x='Adjacency_Type', y=metric, data=summary_pivot, palette='Set2')
            plt.title(f'Average {metric}')
            plt.ylabel(metric)
            plt.xlabel('Adjacency Type')
            for idx, row in summary_pivot.iterrows():
                plt.text(idx % 4, row[metric] + 0.01*row[metric], f"{row[metric]:.4f}",
                         ha='center', va='bottom', fontsize=11)
        plt.tight_layout()
        summary_plot = 'figures/summary_metrics_comparison.png'
        plt.savefig(summary_plot, dpi=300)
        plt.show()  # Changed from plt.close() to plt.show()
        print(f"[Info] Summary metrics comparison plot saved to {summary_plot}")

    plot_summary_metrics(summary_pivot)
