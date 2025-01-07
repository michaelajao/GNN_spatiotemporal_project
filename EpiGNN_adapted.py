"""
EpiGNN Experiments: Static, Dynamic, and Hybrid Adjacency Matrices
==================================================================

This script conducts experiments to compare the performance of the EpiGNN model using
static, dynamic, and hybrid adjacency matrices for forecasting COVID Occupied MV Beds.

Each experiment involves:
1. Training the model with the specified adjacency type.
2. Evaluating the model on the test set.
3. Saving metrics, models, and visualizations.

All results are organized into respective directories for easy comparison.

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
import plotly.express as px
import plotly.io as pio

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import Parameter
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates

# ------------------------------------------------------------------------------
# Set default plot style for Seaborn and Matplotlib
# ------------------------------------------------------------------------------
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# ==============================================================================
# 1. Random Seed & Device Configuration
# ==============================================================================
RANDOM_SEED = 123

def seed_torch(seed=RANDOM_SEED):
    """
    Fix the random seed for reproducibility across numpy, torch, etc.
    """
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
k = 8             # Convolution channels (example)
hidA = 32         # Dimension for Q/K transformations
hidR = 40         # Dimension for the GNN blocks
hidP = 1
n_layer = 4       # Number of GraphConv layers
dropout = 0.5
learning_rate = 1e-3
num_epochs = 1000
batch_size = 32
threshold_distance = 300  # km threshold for adjacency
early_stopping_patience = 10

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
    rolling_features = ['new_confirmed', 'new_deceased', 'newAdmissions',
                        'hospitalCases', 'covidOccupiedMVBeds']

    # 7-day rolling mean per region
    data[rolling_features] = (
        data.groupby('areaName')[rolling_features]
            .rolling(window=7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
    )
    
    # Fill any missing values
    data[rolling_features] = data[rolling_features].fillna(0)
    
    # Ensure data is sorted by areaName and date
    data.sort_values(['areaName', 'date'], inplace=True)
    return data

class NHSRegionDataset(Dataset):
    """
    NHSRegionDataset for sliding-window time-series forecasting.

    - X: (num_timesteps_input, num_nodes, num_features)
    - Y: (num_timesteps_output, num_nodes) -> only the 5th feature: 'covidOccupiedMVBeds'
    """

    def __init__(self, data: pd.DataFrame,
                 num_timesteps_input: int,
                 num_timesteps_output: int,
                 scaler: object = None):
        """
        data: preprocessed DataFrame containing columns:
              ['date', 'areaName', 'latitude', 'longitude', 'population', 'new_confirmed', etc.]
        num_timesteps_input: length of input window
        num_timesteps_output: length of output horizon
        scaler: optional StandardScaler (or similar) for normalization
        """
        super().__init__()
        self.data = data.copy()
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output

        # Extract regions and map to integer indices
        self.regions = self.data['areaName'].unique()
        self.num_nodes = len(self.regions)
        self.region_to_idx = {region: idx for idx, region in enumerate(self.regions)}
        self.data['region_idx'] = self.data['areaName'].map(self.region_to_idx)

        # Define features to keep
        self.features = [
            'new_confirmed', 'new_deceased', 'newAdmissions',
            'hospitalCases', 'covidOccupiedMVBeds'
        ]

        # Pivot: index=date, columns=region_idx, values=features => time-series structure
        self.pivot = self.data.pivot(index='date', columns='region_idx', values=self.features)
        
        # Forward fill missing data by date, fill leftover NaNs with 0
        self.pivot.ffill(inplace=True)
        self.pivot.fillna(0, inplace=True)
        
        # Reshape pivot to (num_dates, num_nodes, num_features)
        self.num_features = len(self.features)
        self.num_dates = self.pivot.shape[0]
        self.feature_array = self.pivot.values.reshape(self.num_dates, self.num_nodes, self.num_features)

        # Optional check for population consistency
        populations = self.data.groupby('areaName')['population'].unique()
        inconsistent_pop = populations[populations.apply(len) > 1]
        if not inconsistent_pop.empty:
            raise ValueError(f"Inconsistent population values in regions: {inconsistent_pop.index.tolist()}")

        # Optional scaling
        if scaler is not None:
            self.scaler = scaler
            # Flatten, scale, reshape
            self.feature_array = self.feature_array.reshape(-1, self.num_features)
            self.feature_array = self.scaler.transform(self.feature_array)
            self.feature_array = self.feature_array.reshape(self.num_dates, self.num_nodes, self.num_features)
        else:
            self.scaler = None

    def __len__(self) -> int:
        """
        Number of samples = total_dates - input_window - output_horizon + 1
        """
        return self.num_dates - self.num_timesteps_input - self.num_timesteps_output + 1

    def __getitem__(self, idx: int):
        """
        Return one sample, containing:
            X: (T_in, num_nodes, num_features)
            Y: (T_out, num_nodes) -> only the 5th feature: 'covidOccupiedMVBeds'
        """
        X = self.feature_array[idx : idx + self.num_timesteps_input]  # shape: (T_in, m, F)
        Y = self.feature_array[idx + self.num_timesteps_input : idx + self.num_timesteps_input + self.num_timesteps_output, :, 4]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def compute_geographic_adjacency(regions: list,
                                 latitudes: list,
                                 longitudes: list,
                                 threshold: float = threshold_distance) -> torch.Tensor:
    """
    Creates a binary adjacency matrix (num_nodes x num_nodes) 
    based on geographic distance threshold using the Haversine formula.
    """

    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Earth radius in km
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
# 5. Model Definition
# ==============================================================================
class GraphConvLayer(nn.Module):
    """
    A basic Graph Convolutional Layer with ELU activation.
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
        # feature: (batch_size, m, in_features)
        # adj:     (batch_size, m, m)
        support = torch.matmul(feature, self.weight)   # => (batch_size, m, out_features)
        output = torch.bmm(adj, support)              # => (batch_size, m, out_features)
        if self.bias is not None:
            return self.act(output + self.bias)
        else:
            return self.act(output)

class GraphLearner(nn.Module):
    """
    Learns an adjacency matrix via a node embedding attention-like mechanism.
    """
    def __init__(self, hidden_dim, tanhalpha=1):
        super(GraphLearner, self).__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

    def forward(self, embedding):
        """
        embedding: (batch_size, m, hidR)
        """
        nodevec1 = torch.tanh(self.alpha * self.linear1(embedding))
        nodevec2 = torch.tanh(self.alpha * self.linear2(embedding))
        adj = (torch.bmm(nodevec1, nodevec2.transpose(1, 2))
               - torch.bmm(nodevec2, nodevec1.transpose(1, 2)))
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))
        return adj

class ConvBranch(nn.Module):
    """
    A single branch of the RegionAwareConv that applies Conv2D + optional pooling
    across the temporal dimension.
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
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            dilation=(dilation_factor, 1)
        )
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.isPool = isPool
        if self.isPool and hidP is not None:
            self.pooling = nn.AdaptiveMaxPool2d((hidP, m))
        self.activate = nn.Tanh()

    def forward(self, x):
        """
        x: (batch_size, in_channels, seq_len, m)
        """
        x = self.conv(x)
        x = self.batchnorm(x)

        if self.isPool and hasattr(self, 'pooling'):
            x = self.pooling(x)  # => (batch_size, out_channels, hidP, m)

        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(-1))  # => (batch_size, out_channels*hidP, m)
        return self.activate(x)

class RegionAwareConv(nn.Module):
    """
    Combines local, period, and global convolution branches for spatiotemporal features.
    """
    def __init__(self, nfeat, P, m, k, hidP, dilation_factor=2):
        super(RegionAwareConv, self).__init__()

        # Local Convs (kernel_size=3,5 with dilation=1)
        self.conv_l1 = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                  kernel_size=3, dilation_factor=1, hidP=hidP)
        self.conv_l2 = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                  kernel_size=5, dilation_factor=1, hidP=hidP)

        # Period Convs (kernel_size=3,5 with larger dilation)
        self.conv_p1 = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                  kernel_size=3, dilation_factor=dilation_factor, hidP=hidP)
        self.conv_p2 = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                  kernel_size=5, dilation_factor=dilation_factor, hidP=hidP)

        # Global Conv (kernel_size=P, no pooling)
        self.conv_g = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                 kernel_size=P, dilation_factor=1, hidP=None,
                                 isPool=False)
        self.activate = nn.Tanh()

    def forward(self, x):
        """
        x: (batch_size, num_features, P, m)
        """
        x_l1 = self.conv_l1(x)
        x_l2 = self.conv_l2(x)
        x_local = torch.cat([x_l1, x_l2], dim=1)

        x_p1 = self.conv_p1(x)
        x_p2 = self.conv_p2(x)
        x_period = torch.cat([x_p1, x_p2], dim=1)

        x_global = self.conv_g(x)

        x = torch.cat([x_local, x_period, x_global], dim=1)
        return self.activate(x).permute(0, 2, 1)  # => (batch_size, m, hidR_something)

class EpiGNN(nn.Module):
    """
    EpiGNN: A spatiotemporal GNN model combining:
      - RegionAwareConv for local/period/global patterns
      - GraphLearner + GraphConv for adjacency inference and message passing
      - Final projection for multi-step forecasting
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

        # Feature extraction backbone
        self.backbone = RegionAwareConv(nfeat=num_features, P=self.w,
                                        m=self.m, k=self.k,
                                        hidP=self.hidP)

        # Q/K transformations
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.t_enc = nn.Linear(1, self.hidR)
        self.s_enc = nn.Linear(1, self.hidR)

        # Gating parameter for adjacency
        self.d_gate = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)
        nn.init.xavier_uniform_(self.d_gate)

        # Graph learner for dynamic adjacency
        self.graphGen = GraphLearner(self.hidR)

        # GNN blocks
        self.GNNBlocks = nn.ModuleList([
            GraphConvLayer(in_features=self.hidR, out_features=self.hidR)
            for _ in range(self.n)
        ])

        # Output projection (concatenate GNN outputs + original embedding)
        self.output = nn.Linear(self.hidR * (self.n) + self.hidR, num_timesteps_output)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                stdv = 1.0 / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, X, adj, adjacency_type='hybrid', states=None, dynamic_adj=None, index=None):
        """
        X:   (batch_size, T, m, F)
        adj: (batch_size, m, m)
        adjacency_type: 'static', 'dynamic', 'hybrid'
        """
        # (1) Permute input to match RegionAwareConv
        X_reshaped = X.permute(0, 3, 1, 2)   # => (batch_size, F, T, m)

        # (2) RegionAwareConv backbone
        temp_emb = self.backbone(X_reshaped) # => (batch_size, m, hidR)

        # (3) Q/K transformations for attention
        query = self.dropout_layer(self.WQ(temp_emb))  # => (batch_size, m, hidA)
        key   = self.dropout_layer(self.WK(temp_emb))  # => (batch_size, m, hidA)

        attn = torch.bmm(query, key.transpose(1, 2))   # => (batch_size, m, m)
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1, keepdim=True)   # => (batch_size, m, 1)
        t_enc = self.dropout_layer(self.t_enc(attn))   # => (batch_size, m, hidR)

        # (4) Local transmission risk
        d = torch.sum(adj, dim=1).unsqueeze(2)         # => (batch_size, m, 1)
        s_enc = self.dropout_layer(self.s_enc(d))      # => (batch_size, m, hidR)

        # (5) Combine embeddings
        feat_emb = temp_emb + t_enc + s_enc            # => (batch_size, m, hidR)

        # (6) Learned adjacency
        learned_adj = self.graphGen(feat_emb)          # => (batch_size, m, m)

        # (7) Handle adjacency types
        if adjacency_type == 'static':
            # Use only static adjacency
            combined_adj = adj
        elif adjacency_type == 'dynamic':
            # Use only dynamic adjacency
            combined_adj = learned_adj
        elif adjacency_type == 'hybrid':
            # Use hybrid adjacency
            d_mat = torch.sum(adj, dim=1, keepdim=True) * torch.sum(adj, dim=2, keepdim=True)
            d_mat = torch.sigmoid(self.d_gate * d_mat)
            spatial_adj = d_mat * adj
            combined_adj = torch.clamp(learned_adj + spatial_adj, 0, 1)
        else:
            raise ValueError("Invalid adjacency_type. Choose from 'static', 'dynamic', 'hybrid'.")

        # (8) Laplacian-like adjacency for GNN
        laplace_adj = getLaplaceMat(X.size(0), self.m, combined_adj)

        # (9) Multi-layer GNN
        node_state = feat_emb
        node_state_list = []
        for layer in self.GNNBlocks:
            node_state = self.dropout_layer(layer(node_state, laplace_adj))
            node_state_list.append(node_state)

        # (10) Concatenate GNN outputs + original embeddings
        node_state_all = torch.cat(node_state_list, dim=-1)
        node_state_all = torch.cat([node_state_all, feat_emb], dim=-1)

        # (11) Final projection => (batch_size, T_out)
        res = self.output(node_state_all)
        return res.transpose(1, 2)  # => (batch_size, T_out, m)

# ==============================================================================
# 6. Data Loading and Normalization
# ==============================================================================
# Create necessary directories
os.makedirs('figures', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

csv_path = "data/merged_nhs_covid_data.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"The specified CSV file does not exist: {csv_path}")

data = pd.read_csv(csv_path, parse_dates=['date'])
data = load_and_correct_data(data, REFERENCE_COORDINATES)

# Create initial dataset (no scaling)
initial_dataset = NHSRegionDataset(data,
                                   num_timesteps_input=num_timesteps_input,
                                   num_timesteps_output=num_timesteps_output,
                                   scaler=None)
print(f"[Info] Total samples in initial dataset: {len(initial_dataset)}")

# Chronological train/val/test split
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

# Create scaled dataset
scaled_dataset = NHSRegionDataset(data,
                                  num_timesteps_input=num_timesteps_input,
                                  num_timesteps_output=num_timesteps_output,
                                  scaler=scaler)
print(f"[Info] Total samples in scaled dataset: {len(scaled_dataset)}")

# Subsets
train_subset = Subset(scaled_dataset, train_indices)
val_subset   = Subset(scaled_dataset, val_indices)
test_subset  = Subset(scaled_dataset, test_indices)

print(f"[Info] Training samples:   {len(train_subset)}")
print(f"[Info] Validation samples: {len(val_subset)}")
print(f"[Info] Test samples:       {len(test_subset)}")

# Dataloaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_subset,   batch_size=batch_size, shuffle=False, drop_last=False)
test_loader  = DataLoader(test_subset,  batch_size=batch_size, shuffle=False, drop_last=False)

# ==============================================================================
# 7. Geographic Adjacency Computation
# ==============================================================================
regions = scaled_dataset.regions.tolist()
latitudes  = [data[data['areaName'] == region]['latitude'].iloc[0]  for region in regions]
longitudes = [data[data['areaName'] == region]['longitude'].iloc[0] for region in regions]

adj_static = compute_geographic_adjacency(regions, latitudes, longitudes).to(device)
print("[Info] Static Adjacency matrix:")
print(adj_static.cpu().numpy())

# Optional: Visualize adjacency as a geographic graph
adj_np = adj_static.cpu().numpy()
G = nx.from_numpy_array(adj_np)
mapping = {i: region for i, region in enumerate(regions)}
G = nx.relabel_nodes(G, mapping)
pos = {region: (longitudes[i], latitudes[i]) for i, region in enumerate(regions)}

plt.figure(figsize=(12, 10))
nx.draw_networkx(G, pos, with_labels=True, node_size=1000,
                 node_color='lightblue', edge_color='gray',
                 font_size=12, font_weight='bold')
plt.title('Geographic Adjacency Graph - Static Adjacency')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.axis('off')
plt.tight_layout()
plt.savefig('figures/geographic_adjacency_graph_static.png', dpi=300)
plt.show()

# ==============================================================================
# 8. Model Initialization
# ==============================================================================
def initialize_model():
    """
    Initialize the EpiGNN model.
    """
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=3,
                                                     verbose=True)
    return model, optimizer, criterion, scheduler

# ==============================================================================
# 9. Training and Evaluation Function
# ==============================================================================
def run_experiment(adjacency_type='hybrid', experiment_id=1, summary_metrics=[]):
    """
    Run training and evaluation for a specific adjacency type.
    
    adjacency_type: 'static', 'dynamic', 'hybrid'
    experiment_id: Identifier for the experiment (1, 2, 3)
    summary_metrics: List to collect metrics for summary
    """
    print(f"\n[Experiment {experiment_id}] Starting with {adjacency_type} adjacency...\n")
    
    # Initialize model, optimizer, criterion, scheduler
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

            batch_size_current = batch_X.size(0)
            
            # Modify adjacency based on experiment
            if adjacency_type == 'static':
                adj_input = adj_static.unsqueeze(0).repeat(batch_size_current, 1, 1)
            elif adjacency_type == 'dynamic':
                adj_input = torch.zeros_like(adj_static).unsqueeze(0).repeat(batch_size_current, 1, 1)
            elif adjacency_type == 'hybrid':
                adj_input = adj_static.unsqueeze(0).repeat(batch_size_current, 1, 1)
            else:
                raise ValueError("Invalid adjacency_type. Choose from 'static', 'dynamic', 'hybrid'.")

            pred = model(batch_X, adj_input, adjacency_type=adjacency_type)
            loss = criterion(pred, batch_Y)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        all_val_preds = []
        all_val_actuals = []

        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                batch_size_current = batch_X.size(0)
                
                # Modify adjacency based on experiment
                if adjacency_type == 'static':
                    adj_input = adj_static.unsqueeze(0).repeat(batch_size_current, 1, 1)
                elif adjacency_type == 'dynamic':
                    adj_input = torch.zeros_like(adj_static).unsqueeze(0).repeat(batch_size_current, 1, 1)
                elif adjacency_type == 'hybrid':
                    adj_input = adj_static.unsqueeze(0).repeat(batch_size_current, 1, 1)
                else:
                    raise ValueError("Invalid adjacency_type. Choose from 'static', 'dynamic', 'hybrid'.")

                pred = model(batch_X, adj_input, adjacency_type=adjacency_type)
                vloss = criterion(pred, batch_Y)
                epoch_val_loss += vloss.item()

                all_val_preds.append(pred.cpu().numpy())
                all_val_actuals.append(batch_Y.cpu().numpy())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Scheduler step
        scheduler.step(avg_val_loss)

        # R² computation per node
        all_val_preds = np.concatenate(all_val_preds, axis=0)     
        all_val_actuals = np.concatenate(all_val_actuals, axis=0) 
        preds_2d   = all_val_preds.reshape(-1, num_nodes)
        actuals_2d = all_val_actuals.reshape(-1, num_nodes)
        r2_vals = r2_score(actuals_2d, preds_2d, multioutput='raw_values')

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val R² (per node): {r2_vals}")

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
    plt.figure(figsize=(12, 8))
    sns.lineplot(x=range(1, len(train_losses) + 1), y=train_losses, label='Train Loss', color='blue')
    sns.lineplot(x=range(1, len(val_losses) + 1), y=val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Training and Validation Loss Curves - {adjacency_type.capitalize()} Adjacency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = f'figures/training_validation_loss_experiment{experiment_id}_{adjacency_type}.png'
    plt.savefig(plot_path, dpi=300)
    plt.show()
    print(f"[Info] Loss curves saved to {plot_path}")

    # Load best model for testing
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Test Evaluation
    test_loss = 0.0
    all_preds = []
    all_actuals = []

    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            batch_size_current = batch_X.size(0)
            
            # Modify adjacency based on experiment
            if adjacency_type == 'static':
                adj_input = adj_static.unsqueeze(0).repeat(batch_size_current, 1, 1)
            elif adjacency_type == 'dynamic':
                adj_input = torch.zeros_like(adj_static).unsqueeze(0).repeat(batch_size_current, 1, 1)
            elif adjacency_type == 'hybrid':
                adj_input = adj_static.unsqueeze(0).repeat(batch_size_current, 1, 1)
            else:
                raise ValueError("Invalid adjacency_type. Choose from 'static', 'dynamic', 'hybrid'.")

            pred = model(batch_X, adj_input, adjacency_type=adjacency_type)
            loss = criterion(pred, batch_Y)
            test_loss += loss.item()

            all_preds.append(pred.cpu())
            all_actuals.append(batch_Y.cpu())

    avg_test_loss = test_loss / len(test_loader)
    print(f"[Experiment {experiment_id}] Test Loss (MSE): {avg_test_loss:.4f}")

    # Combine predictions and actuals
    all_preds = torch.cat(all_preds, dim=0)
    all_actuals = torch.cat(all_actuals, dim=0)

    # Inverse transform only the 'covidOccupiedMVBeds' feature (index 4)
    if scaled_dataset.scaler is not None:
        scale_covid = scaled_dataset.scaler.scale_[4]
        mean_covid  = scaled_dataset.scaler.mean_[4]

        all_preds_np   = all_preds.numpy()   * scale_covid + mean_covid
        all_actuals_np = all_actuals.numpy() * scale_covid + mean_covid
    else:
        all_preds_np   = all_preds.numpy()
        all_actuals_np = all_actuals.numpy()

    # Flatten for final metrics
    preds_flat   = all_preds_np.reshape(-1, num_nodes)
    actuals_flat = all_actuals_np.reshape(-1, num_nodes)

    # Metrics
    mae_per_node = mean_absolute_error(actuals_flat, preds_flat, multioutput='raw_values')
    r2_per_node  = r2_score(actuals_flat, preds_flat, multioutput='raw_values')

    # Save metrics to CSV
    metrics_dict = {
        'Experiment_ID': [],
        'Adjacency_Type': [],
        'Region': [],
        'MAE': [],
        'R2_Score': []
    }

    for idx, region in enumerate(regions):
        metrics_dict['Experiment_ID'].append(experiment_id)
        metrics_dict['Adjacency_Type'].append(adjacency_type)
        metrics_dict['Region'].append(region)
        metrics_dict['MAE'].append(mae_per_node[idx])
        metrics_dict['R2_Score'].append(r2_per_node[idx])

        # Append to summary_metrics
        summary_metrics.append({
            'Experiment_ID': experiment_id,
            'Adjacency_Type': adjacency_type,
            'Region': region,
            'MAE': mae_per_node[idx],
            'R2_Score': r2_per_node[idx]
        })

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_csv_path = f'results/metrics/metrics_experiment{experiment_id}_{adjacency_type}.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"[Info] Metrics saved to {metrics_csv_path}")

    # ==============================================================================
    # 10. Enhanced Visualization
    # ==============================================================================
    unique_dates = data['date'].sort_values().unique()
    forecast_dates = []

    # Map predictions to dates for each sample in the test set
    num_test_samples = len(test_subset)
    for i in range(num_test_samples):
        pred_start_idx = train_size + val_size + i + num_timesteps_input
        pred_end_idx   = pred_start_idx + num_timesteps_output

        if pred_end_idx > len(unique_dates):
            pred_end_idx = len(unique_dates)

        sample_forecast_dates = unique_dates[pred_start_idx : pred_end_idx]

        # If fewer than num_timesteps_output, pad with the last available date
        if len(sample_forecast_dates) < num_timesteps_output:
            last_date = unique_dates[-1]
            sample_forecast_dates = np.append(sample_forecast_dates, [last_date] * (num_timesteps_output - len(sample_forecast_dates)))

        forecast_dates.extend(sample_forecast_dates)

    preds_df = pd.DataFrame(all_preds_np.reshape(-1, num_nodes), columns=regions)
    preds_df['Date'] = forecast_dates

    actuals_df = pd.DataFrame(all_actuals_np.reshape(-1, num_nodes), columns=regions)
    actuals_df['Date'] = forecast_dates

    agg_preds_df  = preds_df.groupby('Date').mean().reset_index()
    agg_actuals_df= actuals_df.groupby('Date').first().reset_index()

    merged_df = pd.merge(agg_preds_df, agg_actuals_df, on='Date', suffixes=('_Predicted', '_Actual'))
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])

    def plot_time_series(region, df, adjacency_type, experiment_id):
        plt.figure(figsize=(14, 7))
        sns.lineplot(x='Date', y=f'{region}_Actual',    data=df, label='Actual', color='blue', marker='o')
        sns.lineplot(x='Date', y=f'{region}_Predicted', data=df, label='Predicted', color='red', linestyle='--', marker='x')
        plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        plt.title(f'Actual vs Predicted COVID Occupied MV Beds for {region} ({adjacency_type.capitalize()} Adjacency)')
        plt.xlabel('Date')
        plt.ylabel('COVID Occupied MV Beds')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = f'figures/actual_vs_predicted_experiment{experiment_id}_{adjacency_type}_{region.replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
        print(f"[Info] Plot saved to {plot_path}")

    for region in regions:
        plot_time_series(region, merged_df, adjacency_type, experiment_id)

    # Additional Visualizations:

    def plot_error_distribution(region, df, adjacency_type, experiment_id):
        errors = df[f'{region}_Predicted'] - df[f'{region}_Actual']
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, bins=30, kde=True, color='purple')
        plt.title(f'Prediction Error Distribution for {region} ({adjacency_type.capitalize()} Adjacency)')
        plt.xlabel('Prediction Error (Predicted - Actual)')
        plt.ylabel('Frequency')
        plt.grid(True)

        mean_error   = errors.mean()
        median_error = errors.median()
        plt.axvline(mean_error,   color='red',   linestyle='dashed', linewidth=1, label=f'Mean: {mean_error:.2f}')
        plt.axvline(median_error, color='green', linestyle='dotted', linewidth=1, label=f'Median: {median_error:.2f}')

        plt.legend()
        plt.tight_layout()
        plot_path = f'figures/error_distribution_experiment{experiment_id}_{adjacency_type}_{region.replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
        print(f"[Info] Error distribution plot saved to {plot_path}")

    for region in regions:
        plot_error_distribution(region, merged_df, adjacency_type, experiment_id)

    def plot_cumulative_error(region, df, adjacency_type, experiment_id):
        errors = df[f'{region}_Predicted'] - df[f'{region}_Actual']
        cumulative_errors = errors.cumsum()
        plt.figure(figsize=(14, 7))
        sns.lineplot(x='Date', y=cumulative_errors, data=df, label='Cumulative Error', color='green')
        plt.fill_between(df['Date'], cumulative_errors, color='green', alpha=0.1)
        plt.title(f'Cumulative Prediction Error Over Time for {region} ({adjacency_type.capitalize()} Adjacency)')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Error')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = f'figures/cumulative_error_experiment{experiment_id}_{adjacency_type}_{region.replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
        print(f"[Info] Cumulative error plot saved to {plot_path}")

    for region in regions:
        plot_cumulative_error(region, merged_df, adjacency_type, experiment_id)

    # Heatmap of Prediction Errors
    for node_idx, region in enumerate(regions):
        errors = all_preds_np[:, :, node_idx] - all_actuals_np[:, :, node_idx]
        plt.figure(figsize=(14, 6))
        sns.heatmap(errors, cmap='coolwarm', annot=False, cbar=True)
        plt.title(f'Heatmap of Prediction Errors for {region} ({adjacency_type.capitalize()} Adjacency)')
        plt.xlabel('Timestep Output')
        plt.ylabel('Sample Index')
        plt.tight_layout()
        plot_path = f'figures/heatmap_error_experiment{experiment_id}_{adjacency_type}_{region.replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
        print(f"[Info] Heatmap of prediction errors saved to {plot_path}")

    def plot_error_boxplot(df, adjacency_type, experiment_id):
        error_data = []
        for region in regions:
            errs = df[f'{region}_Predicted'] - df[f'{region}_Actual']
            error_data.append(pd.Series(errs, name=region))
        
        error_df = pd.concat(error_data, axis=1)
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=error_df, palette='Set2')
        sns.swarmplot(data=error_df, color=".25")
        plt.title(f'Boxplot of Prediction Errors for Each Region ({adjacency_type.capitalize()} Adjacency)')
        plt.xlabel('Region')
        plt.ylabel('Prediction Error (Predicted - Actual)')
        plt.grid(True)
        plt.tight_layout()
        plot_path = f'figures/boxplot_prediction_errors_experiment{experiment_id}_{adjacency_type}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
        print(f"[Info] Boxplot of prediction errors saved to {plot_path}")

    plot_error_boxplot(merged_df, adjacency_type, experiment_id)

    def plot_scatter_actual_vs_predicted(region, df, adjacency_type, experiment_id):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[f'{region}_Actual'], y=df[f'{region}_Predicted'],
                        color='teal', alpha=0.6, edgecolor=None)
        sns.regplot(x=df[f'{region}_Actual'], y=df[f'{region}_Predicted'],
                    scatter=False, color='red', label='Regression Line')
        plt.title(f'Actual vs Predicted COVID Occupied MV Beds for {region} ({adjacency_type.capitalize()} Adjacency)')
        plt.xlabel('Actual COVID Occupied MV Beds')
        plt.ylabel('Predicted COVID Occupied MV Beds')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = f'figures/scatter_actual_vs_predicted_experiment{experiment_id}_{adjacency_type}_{region.replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
        print(f"[Info] Scatter plot saved to {plot_path}")

    for region in regions:
        plot_scatter_actual_vs_predicted(region, merged_df, adjacency_type, experiment_id)

    # ==============================================================================
    # 11. Save Final Model
    # ==============================================================================
    final_model_path = f'models/experiment{experiment_id}/epignn_final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"[Info] Final model saved as '{final_model_path}'.")

    # ==============================================================================
    # 12. Visualize the Learned Adjacency for One Sample
    # ==============================================================================
    def plot_learned_adjacency(model, adj_static, regions, adjacency_type, experiment_id):
        """
        Visualize the combined adjacency (learned + geographic) for a single test sample.
        """
        example_X, _ = next(iter(test_loader))
        example_X = example_X.to(device)
        single_sample_X = example_X[0].unsqueeze(0)  # Shape: (1, T, m, F)
        
        with torch.no_grad():
            # Pass through backbone
            X_reshaped = single_sample_X.permute(0, 3, 1, 2)
            emb_for_adj = model.backbone(X_reshaped)  # (1, m, hidR)
            
            # Pass through GraphLearner
            learned_adj = model.graphGen(emb_for_adj)  # (1, m, m)
            learned_adj = learned_adj.squeeze(0).cpu().numpy()  # (m, m)

            # Handle adjacency type for visualization
            if adjacency_type == 'static':
                combined_adj = adj_static.cpu().numpy()
            elif adjacency_type == 'dynamic':
                combined_adj = learned_adj
            elif adjacency_type == 'hybrid':
                d_mat = np.sum(adj_static.cpu().numpy(), axis=1, keepdims=True) * np.sum(adj_static.cpu().numpy(), axis=0, keepdims=True)
                d_mat = 1 / (1 + np.exp(-model.d_gate.cpu().numpy() * d_mat))  # Sigmoid
                spatial_adj = d_mat * adj_static.cpu().numpy()
                combined_adj = np.clip(learned_adj + spatial_adj, 0, 1)
            else:
                combined_adj = learned_adj  # Default to dynamic

        # Plot adjacency matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(combined_adj, annot=True, fmt=".2f", cmap='viridis',
                    xticklabels=regions, yticklabels=regions)
        plt.title(f'Learned Adjacency Matrix ({adjacency_type.capitalize()} Adjacency) - Experiment {experiment_id}')
        plt.xlabel('Regions')
        plt.ylabel('Regions')
        plt.tight_layout()
        plot_path = f'figures/learned_adjacency_matrix_experiment{experiment_id}_{adjacency_type}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
        print(f"[Info] Learned adjacency matrix plot saved to {plot_path}")

    # Generate adjacency matrix visualization
    plot_learned_adjacency(model, adj_static, regions, adjacency_type, experiment_id)

    print(f"[Experiment {experiment_id}] Workflow complete.\n")

# ==============================================================================
# 10. Run All Experiments
# ==============================================================================
if __name__ == "__main__":
    experiments = [
        {'adjacency_type': 'static', 'experiment_id': 1},
        {'adjacency_type': 'dynamic', 'experiment_id': 2},
        {'adjacency_type': 'hybrid', 'experiment_id': 3},
    ]

    summary_metrics = []

    for exp in experiments:
        run_experiment(adjacency_type=exp['adjacency_type'], experiment_id=exp['experiment_id'], summary_metrics=summary_metrics)

    # ==============================================================================
    # 11. Summary of All Experiments
    # ==============================================================================
    summary_df = pd.DataFrame(summary_metrics)
    summary_csv_path = 'results/metrics/summary_metrics.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[Info] Summary metrics saved to {summary_csv_path}")

    # Pivot the summary dataframe for better readability
    summary_pivot = summary_df.pivot_table(index=['Experiment_ID', 'Adjacency_Type'],
                                          columns='Region',
                                          values=['MAE', 'R2_Score']).reset_index()

    print("\nSummary of All Experiments:")
    print(summary_pivot)

    # Optionally, save the pivot table as well
    summary_pivot_csv_path = 'results/metrics/summary_metrics_pivot.csv'
    summary_pivot.to_csv(summary_pivot_csv_path, index=False)
    print(f"[Info] Summary pivot table saved to {summary_pivot_csv_path}")
