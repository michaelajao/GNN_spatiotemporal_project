#!/usr/bin/env python3
"""
Enhanced EpiGNN Implementation with Advanced Attention and Dynamic Adjacency Updates
=====================================================================================

This script implements an enhanced version of the EpiGNN model for forecasting COVID-19 
occupied mechanical ventilation (MV) beds across UK regions. 

Enhancements include:
1. Advanced Attention Mechanisms: Multi-Head Graph Attention Networks (GATs).
2. Dynamic Adjacency Updates: Adjacency matrices are dynamically learned and updated during training.

Author: [Your Name]
Institution: [Your Institution]
Year: 2025
License: MIT License
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
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

# ------------------------------------------------------------------------------
# Suppress specific warnings (Optional)
# ------------------------------------------------------------------------------
import warnings
from tqdm.auto import tqdm
from tqdm import TqdmWarning

warnings.filterwarnings("ignore", category=TqdmWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# ------------------------------------------------------------------------------
# Set default plot style for Seaborn and Matplotlib
# ------------------------------------------------------------------------------
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# ==============================================================================
# 1. Logging Configuration
# ==============================================================================
import logging

def setup_logging():
    """
    Configures the logging settings.
    Logs are saved to 'experiment.log' and also output to the console.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler("experiment.log"),
            logging.StreamHandler()
        ]
    )

setup_logging()

# ==============================================================================
# 2. Random Seed & Device Configuration
# ==============================================================================
RANDOM_SEED = 123

def seed_everything(seed=RANDOM_SEED):
    """
    Sets the random seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_everything()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# ==============================================================================
# 3. Hyperparameters (Adjusted for Multi-Head GAT)
# ==============================================================================
# Data Parameters
NUM_NODES = 7
NUM_FEATURES = 5  # [new_confirmed, new_deceased, newAdmissions, hospitalCases, covidOccupiedMVBeds]
NUM_TIMESTEPS_INPUT = 14
NUM_TIMESTEPS_OUTPUT = 7

# Model Parameters
K = 8             # Convolution channels (example)
HID_A = 32        # Dimension for Q/K transformations
HID_R = 40        # Dimension for the GNN blocks
HID_P = 1
N_LAYER = 3       # Number of GAT layers
NUM_HEADS = 4     # Number of attention heads
DROPOUT = 0.5

# Training Parameters
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1000
BATCH_SIZE = 32
THRESHOLD_DISTANCE = 300  # km threshold for adjacency
EARLY_STOPPING_PATIENCE = 20

# ==============================================================================
# 4. Reference Coordinates
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
# 5. Data Loading and Preprocessing
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
    logging.info("Data loaded and preprocessed successfully.")
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
        Initializes the NHSRegionDataset.

        Parameters:
        -----------
        data : pd.DataFrame
            The preprocessed data.
        num_timesteps_input : int
            Number of input timesteps.
        num_timesteps_output : int
            Number of output timesteps.
        scaler : StandardScaler or None
            Scaler object for feature normalization.
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
        if 'population' in self.data.columns:
            populations = self.data.groupby('areaName')['population'].unique()
            inconsistent_pop = populations[populations.apply(len) > 1]
            if not inconsistent_pop.empty:
                logging.warning("Inconsistent population data found. Please verify the dataset.")
                self.data = self.data.dropna(subset=['population'])
        else:
            logging.warning("'population' column not found in data.")

        # Optional scaling
        if scaler is not None:
            self.scaler = scaler
            # Flatten, scale, reshape
            self.feature_array = self.feature_array.reshape(-1, self.num_features)
            self.feature_array = self.scaler.fit_transform(self.feature_array)
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

# ==============================================================================
# 5. Model Definitions
# ==============================================================================
# ==============================================================================
# 5a. Multi-Head Graph Attention Layer
# ==============================================================================
class MultiHeadGATLayer(nn.Module):
    """
    A Multi-Head Graph Attention Layer as described in the GAT paper.
    """
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.2, alpha=0.2):
        """
        Args:
            in_features (int): Number of input features per node.
            out_features (int): Number of output features per head.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate.
            alpha (float): Negative slope for LeakyReLU.
        """
        super(MultiHeadGATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_features = out_features

        # Define linear transformations for each head
        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * out_features))
        
        # Define activation functions
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.W.weight.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        """
        Forward pass for the GAT layer.
        
        Args:
            h (torch.Tensor): Node features of shape (batch_size, num_nodes, in_features).
            adj (torch.Tensor): Adjacency matrix of shape (batch_size, num_nodes, num_nodes).
        
        Returns:
            torch.Tensor: Updated node features of shape (batch_size, num_nodes, num_heads * out_features).
        """
        batch_size, num_nodes, _ = h.size()
        Wh = self.W(h)  # (batch_size, num_nodes, num_heads * out_features)
        Wh = Wh.view(batch_size, num_nodes, self.num_heads, self.out_features)  # (batch_size, num_nodes, num_heads, out_features)
        Wh = Wh.permute(0, 2, 1, 3)  # (batch_size, num_heads, num_nodes, out_features)
        
        # Prepare attention
        a_input = torch.cat([Wh.unsqueeze(3).repeat(1, 1, 1, num_nodes, 1),
                             Wh.unsqueeze(2).repeat(1, 1, num_nodes, 1, 1)], dim=-1)  # (batch_size, num_heads, num_nodes, num_nodes, 2*out_features)
        
        # Compute attention scores
        # self.a is (num_heads, 2*out_features)
        # To broadcast, expand self.a to (1, num_heads, 1, 1, 2*out_features)
        a_expanded = self.a.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, num_heads, 1, 1, 2*out_features)
        e = self.leakyrelu((a_input * a_expanded).sum(dim=-1))  # (batch_size, num_heads, num_nodes, num_nodes)
        
        # Masked attention: set to -inf where adj == 0
        e = e.masked_fill(adj.unsqueeze(1) == 0, float('-inf'))
        attention = torch.softmax(e, dim=-1)  # (batch_size, num_heads, num_nodes, num_nodes)
        attention = self.dropout(attention)
        
        # Weighted sum
        h_prime = torch.matmul(attention, Wh)  # (batch_size, num_heads, num_nodes, out_features)
        
        # Concatenate heads
        h_prime = h_prime.permute(0, 2, 1, 3).contiguous()  # (batch_size, num_nodes, num_heads, out_features)
        h_prime = h_prime.view(batch_size, num_nodes, self.num_heads * self.out_features)  # (batch_size, num_nodes, num_heads * out_features)
        
        return h_prime  # (batch_size, num_nodes, num_heads * out_features)

# ==============================================================================
# 5b. GraphLearner
# ==============================================================================
class GraphLearner(nn.Module):
    """
    Learns an adjacency matrix via a node embedding attention-like mechanism.
    Ensures that the learned adjacency includes self-connections to prevent isolated nodes.
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
        
        # Add self-connections to ensure no isolated nodes
        batch_size, m, _ = adj.size()
        identity = torch.eye(m).to(adj.device).unsqueeze(0).repeat(batch_size, 1, 1)
        adj = adj + identity
        
        # Clamp adjacency to prevent extreme values
        adj = torch.clamp(adj, min=1e-6, max=1.0)
        
        return adj

# ==============================================================================
# 5c. ConvBranch and RegionAwareConv
# ==============================================================================
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

# ==============================================================================
# 5d. Updated EpiGNN Model with Multi-Head GAT Layers
# ==============================================================================
class EpiGNN(nn.Module):
    """
    Updated EpiGNN model incorporating Multi-Head Graph Attention Layers for advanced attention mechanisms.
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
                 n_layer=3,
                 num_heads=4,          # Number of attention heads
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

        # Q/K transformations for attention
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.t_enc = nn.Linear(1, self.hidR)
        self.s_enc = nn.Linear(1, self.hidR)

        # Gating parameter for adjacency
        self.d_gate = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)
        nn.init.xavier_uniform_(self.d_gate)

        # Graph learner for dynamic adjacency
        self.graphGen = GraphLearner(self.hidR)

        # Multi-Head GAT Layers
        self.GATLayers = nn.ModuleList([
            MultiHeadGATLayer(in_features=self.hidR, out_features=self.hidR // num_heads, num_heads=num_heads, dropout=dropout)
            for _ in range(self.n)
        ])

        # Output projection (concatenate GAT outputs + original embedding)
        # Calculate the input dimension: (num_heads * out_features * n) + hidR
        gat_output_dim = (self.hidR // num_heads) * num_heads * self.n
        self.output = nn.Linear(gat_output_dim + self.hidR, num_timesteps_output)

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
        Forward pass for EpiGNN.

        Args:
            X (torch.Tensor): Input tensor of shape (batch_size, T, m, F).
            adj (torch.Tensor): Adjacency matrix of shape (batch_size, m, m).
            adjacency_type (str): Type of adjacency to use ('static', 'dynamic', 'hybrid').

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, T_out, m).
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
        d = torch.sum(adj, dim=2).unsqueeze(2)         # => (batch_size, m, 1)
        s_enc = self.dropout_layer(self.s_enc(d))      # => (batch_size, m, hidR)

        # (5) Combine embeddings
        feat_emb = temp_emb + t_enc + s_enc            # => (batch_size, m, hidR)

        # (6) Learned adjacency
        learned_adj = self.graphGen(feat_emb)          # => (batch_size, m, m)

        # (7) Handle adjacency types
        if adjacency_type == 'static':
            combined_adj = adj > 0
        elif adjacency_type == 'dynamic':
            combined_adj = learned_adj > 0
        elif adjacency_type == 'hybrid':
            combined_adj = (learned_adj + adj > 0).float()
        else:
            raise ValueError("Invalid adjacency_type. Choose from 'static', 'dynamic', 'hybrid'.")

        # Ensure combined_adj is binary
        combined_adj = combined_adj.float()

        # (8) Laplacian-like adjacency for GNN
        laplace_adj = getLaplaceMat(X.size(0), self.m, combined_adj)

        # (9) Multi-Head GAT Layers
        node_state = feat_emb
        gat_outputs = []
        for gat in self.GATLayers:
            node_state = gat(node_state, laplace_adj)  # (batch_size, m, num_heads * out_features)
            node_state = F.elu(node_state)
            node_state = self.dropout_layer(node_state)
            gat_outputs.append(node_state)  # Collect outputs for concatenation

        # (10) Concatenate GAT outputs + original embeddings
        gat_concat = torch.cat(gat_outputs, dim=-1)  # (batch_size, m, num_heads * out_features * n)
        node_state_all = torch.cat([gat_concat, feat_emb], dim=-1)  # (batch_size, m, ...)

        # (11) Final projection => (batch_size, T_out, m)
        res = self.output(node_state_all)
        return res.transpose(1, 2)  # => (batch_size, T_out, m)

# ==============================================================================
# 6. Utility Functions
# ==============================================================================
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great-circle distance between two points on the Earth surface.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371  # Radius of Earth in kilometers
    return c * r

from scipy.spatial.distance import pdist, squareform

def compute_geographic_adjacency(regions, latitudes, longitudes, threshold=THRESHOLD_DISTANCE):
    """
    Compute a static adjacency matrix based on geographic distances.
    
    Args:
        regions (list): List of region names.
        latitudes (list): List of latitudes corresponding to regions.
        longitudes (list): List of longitudes corresponding to regions.
        threshold (float): Distance threshold in kilometers to consider adjacency.
    
    Returns:
        torch.Tensor: Adjacency matrix of shape (m, m), where m is the number of regions.
    """
    coords = np.radians(np.column_stack((latitudes, longitudes)))
    # Compute pairwise haversine distances
    def haversine_distance(u, v):
        lat1, lon1 = u
        lat2, lon2 = v
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371  # Earth's radius in km
        return c * r
    
    distances = squareform(pdist(coords, metric=haversine_distance))
    adj_matrix = (distances <= threshold).astype(np.float32)
    np.fill_diagonal(adj_matrix, 1.0)  # Ensure self-connections
    return torch.tensor(adj_matrix, dtype=torch.float32)

def getLaplaceMat(batch_size: int,
                 m: int,
                 adj: torch.Tensor) -> torch.Tensor:
    """
    Compute a Laplacian-like matrix for GCN from adjacency.

    Args:
        batch_size (int): Number of samples in the batch.
        m (int): Number of nodes.
        adj (torch.Tensor): Adjacency matrix of shape (batch_size, m, m).

    Returns:
        torch.Tensor: Normalized adjacency matrix of shape (batch_size, m, m).
    """
    i_mat = torch.eye(m).to(adj.device).unsqueeze(0).expand(batch_size, m, m)
    adj_bin = (adj > 0).float()
    deg = torch.sum(adj_bin, dim=2)  # Degree matrix
    deg_inv = 1.0 / (deg + 1e-12)    # Inverse degree matrix
    deg_inv_mat = i_mat * deg_inv.unsqueeze(2)
    laplace_adj = torch.bmm(deg_inv_mat, adj_bin)  # Normalized adjacency
    return laplace_adj

# ==============================================================================
# 7. Model Initialization
# ==============================================================================
def initialize_model():
    """
    Initialize the updated EpiGNN model with Multi-Head GAT Layers.
    """
    model = EpiGNN(
        num_nodes=NUM_NODES,
        num_features=NUM_FEATURES,
        num_timesteps_input=NUM_TIMESTEPS_INPUT,
        num_timesteps_output=NUM_TIMESTEPS_OUTPUT,
        k=K,
        hidA=HID_A,
        hidR=HID_R,
        hidP=HID_P,
        n_layer=N_LAYER,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
        device=device
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     mode='min',
                                                     factor=0.5,
                                                     patience=3,
                                                     verbose=True)
    logging.info("Model, optimizer, loss function, and scheduler initialized successfully.")
    return model, optimizer, criterion, scheduler

# ==============================================================================
# 8. Training and Evaluation Function
# ==============================================================================
def run_experiment(adjacency_type='hybrid', experiment_id=1, summary_metrics=None,
                  train_loader=None, val_loader=None, test_loader=None,
                  adj_static=None, scaled_dataset=None, regions=None,
                  data=None, test_subset=None, train_size=None, val_size=None):
    """
    Run training and evaluation for a specific adjacency type.

    Parameters:
    -----------
    adjacency_type : str
        Type of adjacency to use ('static', 'dynamic', 'hybrid').
    experiment_id : int
        Identifier for the experiment.
    summary_metrics : list
        List to collect metrics for summary.
    train_loader : DataLoader
        DataLoader for training data.
    val_loader : DataLoader
        DataLoader for validation data.
    test_loader : DataLoader
        DataLoader for test data.
    adj_static : torch.Tensor
        Static adjacency matrix.
    scaled_dataset : NHSRegionDataset
        The scaled dataset containing the scaler.
    regions : list
        List of region names.
    data : pd.DataFrame
        The original preprocessed data.
    test_subset : Subset
        Subset of the dataset used for testing.
    train_size : int
        Number of training samples.
    val_size : int
        Number of validation samples.
    """
    if summary_metrics is None:
        summary_metrics = []

    logging.info(f"Starting Experiment {experiment_id} with {adjacency_type} adjacency.")

    model, optimizer, criterion, scheduler = initialize_model()

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
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
                # Use the learned adjacency from GraphLearner
                adj_input = model.graphGen(model.backbone.forward(batch_X.permute(0, 3, 1, 2)))
            elif adjacency_type == 'hybrid':
                adj_input = adj_static.unsqueeze(0).repeat(batch_size_current, 1, 1) + model.graphGen(model.backbone.forward(batch_X.permute(0, 3, 1, 2)))
            else:
                raise ValueError("Invalid adjacency_type. Choose from 'static', 'dynamic', 'hybrid'.")

            # Ensure adjacency is binary for 'static' and 'hybrid', continuous for 'dynamic'
            if adjacency_type in ['static', 'hybrid']:
                adj_input = (adj_input > 0).float()

            # Forward pass
            pred = model(batch_X, adj_input, adjacency_type=adjacency_type)  # (batch_size, T_out, m)
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
                    adj_input = model.graphGen(model.backbone.forward(batch_X.permute(0, 3, 1, 2)))
                elif adjacency_type == 'hybrid':
                    adj_input = adj_static.unsqueeze(0).repeat(batch_size_current, 1, 1) + model.graphGen(model.backbone.forward(batch_X.permute(0, 3, 1, 2)))
                else:
                    raise ValueError("Invalid adjacency_type. Choose from 'static', 'dynamic', 'hybrid'.")

                # Ensure adjacency is binary for 'static' and 'hybrid', continuous for 'dynamic'
                if adjacency_type in ['static', 'hybrid']:
                    adj_input = (adj_input > 0).float()

                pred = model(batch_X, adj_input, adjacency_type=adjacency_type)
                vloss = criterion(pred, batch_Y)
                epoch_val_loss += vloss.item()

                all_val_preds.append(pred.cpu().numpy())
                all_val_actuals.append(batch_Y.cpu().numpy())

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Compute R² scores
        all_val_preds = np.concatenate(all_val_preds, axis=0)  # (B_total, T_out, m)
        all_val_actuals = np.concatenate(all_val_actuals, axis=0)  # (B_total, T_out, m)
        preds_mean = np.mean(all_val_preds, axis=0)  # (T_out, m)
        actuals_mean = np.mean(all_val_actuals, axis=0)  # (T_out, m)
        r2_vals = []
        for node_idx in range(NUM_NODES):
            if np.isnan(preds_mean[:, node_idx]).any() or np.isnan(actuals_mean[:, node_idx]).any():
                r2 = float('nan')
            else:
                r2 = r2_score(actuals_mean[:, node_idx], preds_mean[:, node_idx])
            r2_vals.append(r2)

        logging.info(f"Epoch {epoch}/{NUM_EPOCHS} - "
                     f"Train Loss: {avg_train_loss:.4f} | "
                     f"Val Loss: {avg_val_loss:.4f} | R² per node: {r2_vals}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_dir = f'models/experiment{experiment_id}_{adjacency_type}'
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = f'{checkpoint_dir}/best_model.pth'
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Saved best model checkpoint at {checkpoint_path}.")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logging.info("Early stopping triggered.")
                break

    # Plot training vs. validation loss
    plt.figure(figsize=(12, 8))
    sns.lineplot(x=range(1, len(train_losses) + 1), y=train_losses, label='Train Loss', color='blue')
    sns.lineplot(x=range(1, len(val_losses) + 1), y=val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Training and Validation Loss Curves - Experiment {experiment_id} ({adjacency_type.capitalize()} Adjacency)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_dir = os.path.join('figures', f'experiment{experiment_id}_{adjacency_type}', 'training_validation_loss')
    os.makedirs(loss_plot_dir, exist_ok=True)
    plot_path = os.path.join(loss_plot_dir, f'training_validation_loss_experiment{experiment_id}_{adjacency_type}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"Loss curves saved to {plot_path}")

    # Load best model for testing
    checkpoint_path = f'models/experiment{experiment_id}_{adjacency_type}/best_model.pth'
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
                adj_input = model.graphGen(model.backbone.forward(batch_X.permute(0, 3, 1, 2)))
            elif adjacency_type == 'hybrid':
                adj_input = adj_static.unsqueeze(0).repeat(batch_size_current, 1, 1) + model.graphGen(model.backbone.forward(batch_X.permute(0, 3, 1, 2)))
            else:
                raise ValueError("Invalid adjacency_type. Choose from 'static', 'dynamic', 'hybrid'.")

            # Ensure adjacency is binary for 'static' and 'hybrid', continuous for 'dynamic'
            if adjacency_type in ['static', 'hybrid']:
                adj_input = (adj_input > 0).float()

            pred = model(batch_X, adj_input, adjacency_type=adjacency_type)
            loss = criterion(pred, batch_Y)
            test_loss += loss.item()

            all_preds.append(pred.cpu().numpy())
            all_actuals.append(batch_Y.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    logging.info(f"Experiment {experiment_id} - Test Loss (MSE): {avg_test_loss:.4f}")

    # Combine predictions and actuals
    all_preds_np = np.concatenate(all_preds, axis=0)      # (B_test, T_out, m)
    all_actuals_np = np.concatenate(all_actuals, axis=0)  # (B_test, T_out, m)

    # Inverse transform only the 'covidOccupiedMVBeds' feature (index 4)
    if scaled_dataset.scaler is not None:
        scale_covid = scaled_dataset.scaler.scale_[4]
        mean_covid  = scaled_dataset.scaler.mean_[4]

        all_preds_np   = all_preds_np * scale_covid + mean_covid
        all_actuals_np = all_actuals_np * scale_covid + mean_covid

    # Flatten for final metrics
    # Reshape to (num_samples, T_out, num_nodes)
    all_preds_reshaped = all_preds_np.reshape(-1, NUM_NODES)   # (num_samples * T_out, num_nodes)
    all_actuals_reshaped = all_actuals_np.reshape(-1, NUM_NODES)
    preds_flat   = all_preds_reshaped.reshape(-1, NUM_NODES)   # (num_samples * T_out, num_nodes)
    actuals_flat = all_actuals_reshaped.reshape(-1, NUM_NODES)

    # Metrics
    mae_per_node = mean_absolute_error(actuals_flat, preds_flat, multioutput='raw_values')
    mse_per_node = mean_squared_error(actuals_flat, preds_flat, multioutput='raw_values')
    rmse_per_node = np.sqrt(mse_per_node)
    r2_per_node  = r2_score(actuals_flat, preds_flat, multioutput='raw_values')
    
    # Calculate Pearson Correlation Coefficient for each region
    pcc_per_node = []
    for i in range(NUM_NODES):
        # Handle cases where variance is zero
        if np.std(actuals_flat[:, i]) == 0 or np.std(preds_flat[:, i]) == 0:
            pcc_per_node.append(0.0)
        else:
            pcc, _ = pearsonr(actuals_flat[:, i], preds_flat[:, i])
            if np.isnan(pcc):
                pcc = 0.0
            pcc_per_node.append(pcc)

    # Save metrics to CSV
    metrics_dict = {
        'Experiment_ID': [],
        'Adjacency_Type': [],
        'Region': [],
        'MAE': [],
        'RMSE': [],
        'R2_Score': [],
        'Pearson_Correlation': []
    }

    for idx, region in enumerate(regions):
        metrics_dict['Experiment_ID'].append(experiment_id)
        metrics_dict['Adjacency_Type'].append(adjacency_type)
        metrics_dict['Region'].append(region)
        metrics_dict['MAE'].append(mae_per_node[idx])
        metrics_dict['RMSE'].append(rmse_per_node[idx])
        metrics_dict['R2_Score'].append(r2_per_node[idx])
        metrics_dict['Pearson_Correlation'].append(pcc_per_node[idx])

        summary_metrics.append({
            'Experiment_ID': experiment_id,
            'Adjacency_Type': adjacency_type,
            'Region': region,
            'MAE': mae_per_node[idx],
            'RMSE': rmse_per_node[idx],
            'R2_Score': r2_per_node[idx],
            'Pearson_Correlation': pcc_per_node[idx]
        })

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_csv_dir = os.path.join('results', 'metrics', f'experiment{experiment_id}_{adjacency_type}')
    os.makedirs(metrics_csv_dir, exist_ok=True)
    metrics_csv_path = os.path.join(metrics_csv_dir, f'metrics_experiment{experiment_id}_{adjacency_type}.csv')
    metrics_df.to_csv(metrics_csv_path, index=False)
    logging.info(f"Metrics saved to {metrics_csv_path}")

    # ==============================================================================
    # 9. Visualization
    # ==============================================================================
    def plot_actual_vs_predicted(preds, actuals, regions, experiment_id, adjacency_type, output_dir='figures'):
        """
        Plot Actual vs. Predicted Time Series for each region.
        """
        full_output_dir = os.path.join(output_dir, f'experiment{experiment_id}_{adjacency_type}', 'actual_vs_predicted')
        os.makedirs(full_output_dir, exist_ok=True)
        
        time_steps = range(NUM_TIMESTEPS_OUTPUT)

        # Compute mean over samples for plotting
        preds_mean = np.mean(preds.reshape(-1, NUM_TIMESTEPS_OUTPUT, NUM_NODES), axis=0)  # (T_out, m)
        actuals_mean = np.mean(actuals.reshape(-1, NUM_TIMESTEPS_OUTPUT, NUM_NODES), axis=0)  # (T_out, m)

        for region_idx, region in enumerate(regions):
            plt.figure(figsize=(12, 6))
            plt.plot(time_steps, actuals_mean[:, region_idx], label='Actual', marker='o')
            plt.plot(time_steps, preds_mean[:, region_idx], label='Predicted', marker='x')
            plt.title(f'Actual vs. Predicted COVID Occupied MV Beds - {region}')
            plt.xlabel('Time Steps')
            plt.ylabel('COVID Occupied MV Beds')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(full_output_dir, f'actual_vs_predicted_experiment{experiment_id}_{adjacency_type}_{region}.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            logging.info(f"Actual vs. Predicted plot saved to {plot_path}")

    def plot_error_distribution(preds, actuals, regions, experiment_id, adjacency_type, output_dir='figures'):
        """
        Plot error distribution histograms for each region.
        """
        full_output_dir = os.path.join(output_dir, f'experiment{experiment_id}_{adjacency_type}', 'error_distributions')
        os.makedirs(full_output_dir, exist_ok=True)
        
        errors = preds - actuals

        for region_idx, region in enumerate(regions):
            plt.figure(figsize=(8, 6))
            sns.histplot(errors[:, region_idx], bins=30, kde=True, color='skyblue')
            plt.title(f'Error Distribution - {region}')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(full_output_dir, f'error_distribution_experiment{experiment_id}_{adjacency_type}_{region}.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            logging.info(f"Error distribution plot saved to {plot_path}")

    def plot_cumulative_error(preds, actuals, regions, experiment_id, adjacency_type, output_dir='figures'):
        """
        Plot cumulative errors over time for each region.
        """
        full_output_dir = os.path.join(output_dir, f'experiment{experiment_id}_{adjacency_type}', 'cumulative_errors')
        os.makedirs(full_output_dir, exist_ok=True)
        
        errors = preds - actuals
        cumulative_errors = np.cumsum(errors, axis=0)

        for region_idx, region in enumerate(regions):
            plt.figure(figsize=(12, 6))
            plt.plot(cumulative_errors[:, region_idx], label='Cumulative Error', color='red')
            plt.title(f'Cumulative Error over Time - {region}')
            plt.xlabel('Time Steps')
            plt.ylabel('Cumulative Error')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(full_output_dir, f'cumulative_error_experiment{experiment_id}_{adjacency_type}_{region}.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            logging.info(f"Cumulative error plot saved to {plot_path}")

    def plot_error_heatmap(preds, actuals, regions, experiment_id, adjacency_type, output_dir='figures'):
        """
        Plot heatmaps of prediction errors for each region.
        """
        full_output_dir = os.path.join(output_dir, f'experiment{experiment_id}_{adjacency_type}', 'error_heatmaps')
        os.makedirs(full_output_dir, exist_ok=True)
        
        # Reshape the predictions and actuals to 2D
        preds_2d = preds.reshape(-1, len(regions))    # (num_samples * num_timesteps, num_regions)
        actuals_2d = actuals.reshape(-1, len(regions)) # (num_samples * num_timesteps, num_regions)
        errors = preds_2d - actuals_2d                 # (num_samples * num_timesteps, num_regions)

        # Create heatmap for all regions
        plt.figure(figsize=(15, 8))
        sns.heatmap(errors, annot=False, fmt=".2f", cmap='coolwarm', 
                    xticklabels=regions, yticklabels=False)
        plt.title(f'Prediction Errors Heatmap - All Regions')
        plt.xlabel('Regions')
        plt.ylabel('Time Steps')
        plt.tight_layout()
        plot_path = os.path.join(full_output_dir, f'error_heatmap_experiment{experiment_id}_{adjacency_type}_all_regions.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"Error heatmap saved to {plot_path}")

    def plot_error_boxplot(preds, actuals, regions, experiment_id, adjacency_type, output_dir='figures'):
        """
        Plot boxplots of prediction errors for each region.
        """
        full_output_dir = os.path.join(output_dir, f'experiment{experiment_id}_{adjacency_type}', 'error_boxplots')
        os.makedirs(full_output_dir, exist_ok=True)
        
        errors = preds - actuals

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=errors, orient='h', palette='Set2')
        plt.title(f'Prediction Errors Boxplot - {adjacency_type.capitalize()} Adjacency')
        plt.xlabel('Prediction Error')
        plt.yticks(ticks=range(NUM_NODES), labels=regions)
        plt.tight_layout()
        plot_path = os.path.join(full_output_dir, f'error_boxplot_experiment{experiment_id}_{adjacency_type}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"Error boxplot saved to {plot_path}")

    def plot_scatter_actual_vs_predicted(preds, actuals, regions, experiment_id, adjacency_type, output_dir='figures'):
        """
        Plot scatter plots of Actual vs Predicted values for each region.
        """
        full_output_dir = os.path.join(output_dir, f'experiment{experiment_id}_{adjacency_type}', 'scatter_plots')
        os.makedirs(full_output_dir, exist_ok=True)
        
        for region_idx, region in enumerate(regions):
            plt.figure(figsize=(8, 8))
            sns.scatterplot(x=actuals[:, region_idx], y=preds[:, region_idx], color='purple', alpha=0.6)
            plt.plot([actuals[:, region_idx].min(), actuals[:, region_idx].max()],
                     [actuals[:, region_idx].min(), actuals[:, region_idx].max()],
                     color='red', linestyle='--', label='Ideal')
            plt.title(f'Actual vs. Predicted COVID Occupied MV Beds - {region}')
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_path = os.path.join(full_output_dir, f'scatter_actual_vs_predicted_experiment{experiment_id}_{adjacency_type}_{region}.png')
            plt.savefig(plot_path, dpi=300)
            plt.close()
            logging.info(f"Scatter plot saved to {plot_path}")

    def plot_error_metrics_heatmap(summary_df, experiment_id, adjacency_type, output_dir='figures'):
        """
        Plot a heatmap of error metrics (MAE, RMSE, R2_Score, Pearson_Correlation) across regions.
        """
        full_output_dir = os.path.join(output_dir, f'experiment{experiment_id}_{adjacency_type}', 'metrics_heatmap')
        os.makedirs(full_output_dir, exist_ok=True)
        
        metrics = ['MAE', 'RMSE', 'R2_Score', 'Pearson_Correlation']
        heatmap_data = summary_df[
            (summary_df['Experiment_ID'] == experiment_id) &
            (summary_df['Adjacency_Type'] == adjacency_type)
        ].set_index('Region')[metrics]

        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap='YlGnBu')
        plt.title(f'Error Metrics Heatmap - Experiment {experiment_id} ({adjacency_type.capitalize()} Adjacency)')
        plt.xlabel('Metrics')
        plt.ylabel('Regions')
        plt.tight_layout()
        plot_path = os.path.join(full_output_dir, f'metrics_heatmap_experiment{experiment_id}_{adjacency_type}.png')
        plt.savefig(plot_path, dpi=300)
        plt.close()
        logging.info(f"Metrics heatmap saved to {plot_path}")

    # Generate All Visualizations
    logging.info("Generating additional visualizations...")

    plot_actual_vs_predicted(all_preds_np, all_actuals_np, regions, experiment_id, adjacency_type)
    plot_error_distribution(all_preds_np, all_actuals_np, regions, experiment_id, adjacency_type)
    plot_cumulative_error(all_preds_np, all_actuals_np, regions, experiment_id, adjacency_type)
    plot_error_heatmap(all_preds_np, all_actuals_np, regions, experiment_id, adjacency_type)
    plot_error_boxplot(all_preds_np, all_actuals_np, regions, experiment_id, adjacency_type)
    plot_scatter_actual_vs_predicted(all_preds_np, all_actuals_np, regions, experiment_id, adjacency_type)
    plot_error_metrics_heatmap(pd.DataFrame(summary_metrics), experiment_id, adjacency_type)
    logging.info("All additional visualizations generated and saved.")

    # ==============================================================================
    # 10. Save Final Model
    # ==============================================================================
    final_model_path = f'models/experiment{experiment_id}_{adjacency_type}/epignn_final_model.pth'
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved as '{final_model_path}'.")

    logging.info(f"Experiment {experiment_id} workflow complete.\n")

# ==============================================================================
# 9. Main Execution: Run All Experiments
# ==============================================================================
def main():
    """
    Main function to execute experiments.
    """
    # Setup directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    logging.info("Directories set up successfully.")

    # Load data
    csv_path = "data/merged_nhs_covid_data.csv"
    if not os.path.exists(csv_path):
        logging.error(f"The specified CSV file does not exist: {csv_path}")
        raise FileNotFoundError(f"The specified CSV file does not exist: {csv_path}")

    data = pd.read_csv(csv_path, parse_dates=['date'])
    data = load_and_correct_data(data, REFERENCE_COORDINATES)

    # Create initial dataset (no scaling)
    initial_dataset = NHSRegionDataset(data,
                                       num_timesteps_input=NUM_TIMESTEPS_INPUT,
                                       num_timesteps_output=NUM_TIMESTEPS_OUTPUT,
                                       scaler=None)
    logging.info(f"Total samples in initial dataset: {len(initial_dataset)}")

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

    train_features = np.concatenate(train_features, axis=0).reshape(-1, NUM_FEATURES)
    scaler.fit(train_features)
    logging.info("Scaler fitted on training data.")

    # Create scaled dataset
    scaled_dataset = NHSRegionDataset(data,
                                      num_timesteps_input=NUM_TIMESTEPS_INPUT,
                                      num_timesteps_output=NUM_TIMESTEPS_OUTPUT,
                                      scaler=scaler)
    logging.info(f"Total samples in scaled dataset: {len(scaled_dataset)}")

    # Subsets
    train_subset = Subset(scaled_dataset, train_indices)
    val_subset   = Subset(scaled_dataset, val_indices)
    test_subset  = Subset(scaled_dataset, test_indices)

    logging.info(f"Training samples:   {len(train_subset)}")
    logging.info(f"Validation samples: {len(val_subset)}")
    logging.info(f"Test samples:       {len(test_subset)}")

    # Dataloaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_subset,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # ==============================================================================
    # 10. Geographic Adjacency Computation
    # ==============================================================================
    regions = scaled_dataset.regions.tolist()
    latitudes  = [data[data['areaName'] == region]['latitude'].iloc[0]  for region in regions]
    longitudes = [data[data['areaName'] == region]['longitude'].iloc[0] for region in regions]

    adj_static = compute_geographic_adjacency(regions, latitudes, longitudes, THRESHOLD_DISTANCE).to(device)
    logging.info("Static Adjacency Matrix:")
    logging.info(adj_static.cpu().numpy())

    # Visualize adjacency as a geographic graph
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
    plot_path = 'figures/geographic_adjacency_graph_static.png'
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logging.info(f"Geographic adjacency graph saved to {plot_path}")

    # ==============================================================================
    # 11. Run All Experiments
    # ==============================================================================
    experiments = [
        {'adjacency_type': 'static', 'experiment_id': 1},
        {'adjacency_type': 'dynamic', 'experiment_id': 2},
        {'adjacency_type': 'hybrid', 'experiment_id': 3},
    ]

    summary_metrics = []

    for exp in experiments:
        run_experiment(
            adjacency_type=exp['adjacency_type'],
            experiment_id=exp['experiment_id'],
            summary_metrics=summary_metrics,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            adj_static=adj_static,
            scaled_dataset=scaled_dataset,
            regions=regions,
            data=data,
            test_subset=test_subset,
            train_size=train_size,
            val_size=val_size
        )

    # ==============================================================================
    # 12. Summary of All Experiments
    # ==============================================================================
    summary_df = pd.DataFrame(summary_metrics)
    summary_csv_path = 'results/metrics/summary_metrics.csv'
    os.makedirs(os.path.dirname(summary_csv_path), exist_ok=True)
    summary_df.to_csv(summary_csv_path, index=False)
    logging.info(f"Summary metrics saved to {summary_csv_path}")

    # Pivot the summary dataframe for better readability
    summary_pivot = summary_df.pivot_table(index=['Experiment_ID', 'Adjacency_Type'],
                                          columns='Region',
                                          values=['MAE', 'RMSE', 'R2_Score', 'Pearson_Correlation']).reset_index()

    logging.info("Summary of All Experiments:")
    logging.info(summary_pivot)

    summary_pivot_csv_path = 'results/metrics/summary_metrics_pivot.csv'
    summary_pivot.to_csv(summary_pivot_csv_path, index=False)
    logging.info(f"Summary pivot table saved to {summary_pivot_csv_path}")

    # Optional: Create a summary plot comparing different adjacency types
    def plot_summary_metrics_comparison(summary_pivot: pd.DataFrame):
        """
        Plots average metrics across different adjacency types.

        Parameters:
        -----------
        summary_pivot : pd.DataFrame
            Pivoted summary metrics DataFrame.
        """
        metrics_list = ['MAE', 'RMSE', 'R2_Score', 'Pearson_Correlation']
        plt.figure(figsize=(20, 12))
        
        # Determine grid size based on number of metrics
        num_metrics = len(metrics_list)
        cols = 2
        rows = math.ceil(num_metrics / cols)
        
        for i, metric in enumerate(metrics_list):
            plt.subplot(rows, cols, i+1)
            
            # Compute mean metric across regions
            metric_mean = summary_pivot.groupby('Adjacency_Type')[metric].mean().reset_index()
            
            sns.barplot(x='Adjacency_Type', y=metric, data=metric_mean, palette='Set2')
            
            plt.title(f'Average {metric} across Adjacency Types')
            plt.xlabel('Adjacency Type')
            plt.ylabel(metric)
            
            # Dynamic y-axis limits
            min_val = summary_pivot[metric].min()
            max_val = summary_pivot[metric].max()
            if metric in ['R2_Score', 'Pearson_Correlation']:
                plt.ylim(min(-1, min_val * 1.2), max(1, max_val * 1.2))
            else:
                plt.ylim(0, max(max_val, 1) * 1.2)
            
            # Text annotations with safe y positions
            for idx, row in metric_mean.iterrows():
                x_pos = idx
                y_pos = row[metric] + 0.05 * max_val
                plt.text(x_pos, y_pos, f"{row[metric]:.4f}",
                        ha='center', va='bottom', fontsize=10)
            
        plt.tight_layout()
        summary_plot = 'figures/summary_metrics_comparison.png'
        plt.savefig(summary_plot, dpi=300)
        plt.close()
        logging.info(f"Summary metrics comparison plot saved to {summary_plot}")

    plot_summary_metrics_comparison(summary_pivot)

if __name__ == "__main__":
    main()
