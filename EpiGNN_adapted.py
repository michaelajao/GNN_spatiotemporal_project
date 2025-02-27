#!/usr/bin/env python3
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import Parameter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import matplotlib.dates as mdates

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
# 3. Hyperparameters and Configuration
# ==============================================================================
# Data Parameters
NUM_NODES = 7
NUM_FEATURES = 5  # [new_confirmed, new_deceased, newAdmissions, hospitalCases, covidOccupiedMVBeds]
NUM_TIMESTEPS_INPUT = 14
NUM_TIMESTEPS_OUTPUT = 7

# Model Parameters
K = 8          # Convolution channels in RegionAwareConv
HID_A = 32     # Dimension for Q/K transformations (for adjacency learning)
HID_R = 40     # Dimension for node embeddings in the GNN blocks
HID_P = 1
N_LAYER = 3    # Number of GraphAttn layers
DROPOUT = 0.5

# Training Parameters
LEARNING_RATE = 1e-4
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
    Assign correct (latitude, longitude) for each region, apply a 7-day rolling 
    average to certain features, and sort chronologically.

    Parameters:
    -----------
    data : pd.DataFrame
        Raw data.
    reference_coordinates : dict
        Mapping of region names to their (latitude, longitude).

    Returns:
    --------
    data : pd.DataFrame
        Preprocessed data.
    """
    # Assign correct geographic coordinates
    for region, coords in reference_coordinates.items():
        data.loc[data['areaName'] == region, ['latitude', 'longitude']] = coords

    # Define features for 7-day rolling
    rolling_features = ['new_confirmed', 'new_deceased',
                        'newAdmissions', 'hospitalCases', 'covidOccupiedMVBeds']

    # Apply 7-day rolling mean per region
    data[rolling_features] = (
        data.groupby('areaName')[rolling_features]
            .rolling(window=7, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
    )

    # Fill missing values with 0
    data[rolling_features] = data[rolling_features].fillna(0)

    # Sort by region and date
    data.sort_values(['areaName', 'date'], inplace=True)
    logging.info("Data loaded and preprocessed successfully.")
    return data

class NHSRegionDataset(Dataset):
    """
    NHSRegionDataset for sliding-window time-series forecasting.

    Attributes:
    -----------
    data : pd.DataFrame
        The preprocessed data.
    num_timesteps_input : int
        Number of input timesteps.
    num_timesteps_output : int
        Number of output timesteps.
    scaler : StandardScaler or None
        Scaler object for feature normalization.
    regions : list
        List of unique regions.
    num_nodes : int
        Number of regions/nodes.
    region_to_idx : dict
        Mapping from region names to indices.
    features : list
        List of feature names.
    pivot : pd.DataFrame
        Pivoted dataframe indexed by date and columns as region-feature.
    feature_array : np.ndarray
        Numpy array of shape (num_dates, num_nodes, num_features).
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
            The raw data.
        num_timesteps_input : int
            Number of input timesteps.
        num_timesteps_output : int
            Number of output timesteps.
        scaler : StandardScaler or None
            Scaler for feature normalization.
        """
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
        if 'population' in self.data.columns:
            populations = self.data.groupby('areaName')['population'].unique()
            inconsistent_pop = populations[populations.apply(len) > 1]
            if not inconsistent_pop.empty:
                # Handle inconsistent population data
                logging.warning("Inconsistent population data found. Please verify the dataset.")
                self.data = self.data.dropna(subset=['population'])
        else:
            logging.warning("'population' column not found in data.")

        # Optional scaling
        if scaler is not None:
            self.scaler = scaler
            # Reshape for scaling: (num_dates * num_nodes, num_features)
            self.feature_array = self.scaler.fit_transform(self.feature_array.reshape(-1, self.num_features)).reshape(self.num_dates, self.num_nodes, self.num_features)
        else:
            self.scaler = None

    def __len__(self) -> int:
        return self.num_dates - self.num_timesteps_input - self.num_timesteps_output + 1

    def __getitem__(self, idx: int):
        X = self.feature_array[idx : idx + self.num_timesteps_input]  # shape: (T_in, m, F)
        Y = self.feature_array[idx + self.num_timesteps_input : idx + self.num_timesteps_input + self.num_timesteps_output, :, 4]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# ==============================================================================
# 6. Adjacency Matrix Computation
# ==============================================================================
def compute_geographic_adjacency(regions: list,
                                 latitudes: list,
                                 longitudes: list,
                                 threshold: float = THRESHOLD_DISTANCE) -> torch.Tensor:
    """
    Creates a binary adjacency matrix based on distance threshold using Haversine formula.

    Parameters:
    -----------
    regions : list
        List of region names.
    latitudes : list
        List of latitudes corresponding to regions.
    longitudes : list
        List of longitudes corresponding to regions.
    threshold : float
        Distance threshold in kilometers for adjacency.

    Returns:
    --------
    adj_matrix : torch.Tensor
        Binary adjacency matrix of shape (num_nodes, num_nodes).
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
        for j in range(i, num_nodes):  # Ensure symmetry and avoid redundant calculations
            if i == j:
                adj_matrix[i][j] = 1
            else:
                distance = haversine(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                if distance <= threshold:
                    adj_matrix[i][j] = 1
                    adj_matrix[j][i] = 1  # Ensure symmetry

    logging.info("Geographic adjacency matrix computed successfully.")
    return torch.tensor(adj_matrix, dtype=torch.float32)

def get_laplacian_matrix(batch_size: int,
                         num_nodes: int,
                         adj: torch.Tensor) -> torch.Tensor:
    """
    Computes a normalized adjacency matrix (similar to Laplacian) for GCN.

    Parameters:
    -----------
    batch_size : int
        Number of samples in the batch.
    num_nodes : int
        Number of nodes/regions.
    adj : torch.Tensor
        Adjacency matrix of shape (batch_size, num_nodes, num_nodes).

    Returns:
    --------
    laplace_adj : torch.Tensor
        Normalized adjacency matrix of shape (batch_size, num_nodes, num_nodes).
    """
    i_mat = torch.eye(num_nodes).to(adj.device).unsqueeze(0).repeat(batch_size, 1, 1)
    adj_bin = (adj > 0).float()
    deg = torch.sum(adj_bin, dim=2)  # Degree matrix
    deg_inv = 1.0 / (deg + 1e-12)    # Inverse degree matrix
    deg_inv_mat = i_mat * deg_inv.unsqueeze(2)
    laplace_adj = torch.bmm(deg_inv_mat, adj_bin)  # Normalized adjacency
    return laplace_adj

# ==============================================================================
# 7. Model Definitions: ConvBranch, RegionAwareConv, GraphAttnLayer, GraphLearner, EpiGNN
# ==============================================================================
class ConvBranch(nn.Module):
    """
    Single branch of RegionAwareConv that applies Conv2D + optional pooling
    across the temporal dimension.
    """
    def __init__(self, m: int, in_channels: int, out_channels: int, kernel_size: int,
                 dilation_factor: int = 2, hidP: int = 1, isPool: bool = True):
        """
        Initializes the ConvBranch.

        Parameters:
        -----------
        m : int
            Number of regions/nodes.
        in_channels : int
            Number of input channels/features.
        out_channels : int
            Number of output channels/features.
        kernel_size : int
            Kernel size for convolution.
        dilation_factor : int, optional
            Dilation factor for convolution, by default 2.
        hidP : int, optional
            Pooling size parameter, by default 1.
        isPool : bool, optional
            Whether to apply pooling, by default True.
        """
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
        """
        Forward pass for ConvBranch.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, T, m).

        Returns:
        --------
        torch.Tensor
            Activated tensor after convolution and pooling.
        """
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.isPool and hasattr(self, 'pooling'):
            x = self.pooling(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1, x.size(-1))
        return self.activate(x)

class RegionAwareConv(nn.Module):
    """
    RegionAwareConv: local, period, global branches.
    """
    def __init__(self, nfeat: int, P: int, m: int, k: int, hidP: int, dilation_factor: int = 2):
        """
        Initializes the RegionAwareConv module.

        Parameters:
        -----------
        nfeat : int
            Number of input features.
        P : int
            Kernel size for the global branch.
        m : int
            Number of regions/nodes.
        k : int
            Number of convolution channels per branch.
        hidP : int
            Pooling size parameter.
        dilation_factor : int, optional
            Dilation factor for convolution, by default 2.
        """
        super(RegionAwareConv, self).__init__()
        self.conv_l1 = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                  kernel_size=3, dilation_factor=1, hidP=hidP)
        self.conv_l2 = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                  kernel_size=5, dilation_factor=1, hidP=hidP)
        self.conv_p1 = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                  kernel_size=3, dilation_factor=dilation_factor, hidP=hidP)
        self.conv_p2 = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                  kernel_size=5, dilation_factor=dilation_factor, hidP=hidP)
        self.conv_g = ConvBranch(m=m, in_channels=nfeat, out_channels=k,
                                 kernel_size=P, dilation_factor=1, hidP=None, isPool=False)
        self.activate = nn.Tanh()

    def forward(self, x):
        """
        Forward pass for RegionAwareConv.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, nfeat, T, m).

        Returns:
        --------
        torch.Tensor
            Activated tensor after concatenation of all branches.
        """
        x_l1 = self.conv_l1(x)
        x_l2 = self.conv_l2(x)
        x_local = torch.cat([x_l1, x_l2], dim=1)

        x_p1 = self.conv_p1(x)
        x_p2 = self.conv_p2(x)
        x_period = torch.cat([x_p1, x_p2], dim=1)

        x_global = self.conv_g(x)
        x = torch.cat([x_local, x_period, x_global], dim=1)
        return self.activate(x).permute(0, 2, 1)  # => (batch_size, m, k*3)

class GraphAttnLayer(nn.Module):
    """
    Spatiotemporal Graph Attention Layer.
    Expects input shape: (batch_size, m, in_features).
    """
    def __init__(self, in_features: int, out_features: int, alpha: float = 0.2):
        """
        Initializes the GraphAttnLayer.

        Parameters:
        -----------
        in_features : int
            Number of input features per node.
        out_features : int
            Number of output features per node.
        alpha : float, optional
            Negative slope for LeakyReLU, by default 0.2.
        """
        super(GraphAttnLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        # Weight matrix for node features
        self.W = Parameter(torch.Tensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W)

        # Attention coefficients a
        self.a = Parameter(torch.Tensor(2 * out_features, 1))
        nn.init.xavier_uniform_(self.a)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GraphAttnLayer.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, m, in_features).
        adj : torch.Tensor
            Adjacency matrix of shape (batch_size, m, m).

        Returns:
        --------
        torch.Tensor
            Output tensor after graph attention.
        """
        B, M, _ = x.size()

        # Linear transformation
        h = torch.matmul(x, self.W)  # (B, m, out_features)

        # Compute attention scores
        h_i = h.unsqueeze(2).repeat(1, 1, M, 1)  # (B, m, m, out_features)
        h_j = h.unsqueeze(1).repeat(1, M, 1, 1)  # (B, m, m, out_features)
        a_input = torch.cat([h_i, h_j], dim=-1)  # (B, m, m, 2*out_features)

        e = torch.matmul(a_input, self.a).squeeze(-1)  # (B, m, m)
        e = self.leakyrelu(e)

        # Masked attention: only consider adjacent nodes
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)

        # Softmax normalization
        alpha_attn = F.softmax(attention, dim=-1)  # (B, m, m)

        # Weighted sum of neighbor features
        h_prime = torch.bmm(alpha_attn, h)  # (B, m, out_features)

        return F.elu(h_prime)

class GraphLearner(nn.Module):
    """
    Learns dynamic adjacency matrix based on node embeddings.
    """
    def __init__(self, hidden_dim: int, tanhalpha: float = 1):
        """
        Initializes the GraphLearner.

        Parameters:
        -----------
        hidden_dim : int
            Dimension of node embeddings.
        tanhalpha : float, optional
            Scaling factor for tanh activation, by default 1.
        """
        super(GraphLearner, self).__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for GraphLearner.

        Parameters:
        -----------
        embedding : torch.Tensor
            Node embeddings of shape (batch_size, m, hidR).

        Returns:
        --------
        torch.Tensor
            Learned adjacency matrix of shape (batch_size, m, m).
        """
        nodevec1 = torch.tanh(self.alpha * self.linear1(embedding))
        nodevec2 = torch.tanh(self.alpha * self.linear2(embedding))
        adj = (torch.bmm(nodevec1, nodevec2.transpose(1, 2))
               - torch.bmm(nodevec2, nodevec1.transpose(1, 2)))
        adj = self.alpha * adj
        adj = torch.relu(torch.tanh(adj))
        return adj

class EpiGNN(nn.Module):
    """
    EpiGNN: A spatiotemporal GNN model combining:
      - RegionAwareConv for local/period/global patterns
      - GraphAttnLayer for graph-based message passing
      - Final projection for multi-step forecasting
    """
    def __init__(self,
                 num_nodes: int,
                 num_features: int,
                 num_timesteps_input: int,
                 num_timesteps_output: int,
                 k: int = K,
                 hidA: int = HID_A,
                 hidR: int = HID_R,
                 hidP: int = HID_P,
                 n_layer: int = N_LAYER,
                 dropout: float = DROPOUT,
                 device: str = 'cpu'):
        """
        Initializes the EpiGNN model.

        Parameters:
        -----------
        num_nodes : int
            Number of regions/nodes.
        num_features : int
            Number of input features.
        num_timesteps_input : int
            Number of input timesteps.
        num_timesteps_output : int
            Number of output timesteps.
        k : int, optional
            Number of convolution channels per branch, by default 8.
        hidA : int, optional
            Dimension for Q/K transformations, by default 32.
        hidR : int, optional
            Dimension for node embeddings in GNN blocks, by default 40.
        hidP : int, optional
            Pooling size parameter, by default 1.
        n_layer : int, optional
            Number of GraphAttn layers, by default 3.
        dropout : float, optional
            Dropout rate, by default 0.5.
        device : str, optional
            Device to run the model on, by default 'cpu'.
        """
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

        # RegionAwareConv as backbone
        self.backbone = RegionAwareConv(nfeat=num_features, P=self.w,
                                        m=self.m, k=self.k, hidP=self.hidP)

        # Q/K transformations for adjacency learning
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.t_enc = nn.Linear(1, self.hidR)
        self.s_enc = nn.Linear(1, self.hidR)

        # Gate parameter for hybrid adjacency
        self.d_gate = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)
        nn.init.xavier_uniform_(self.d_gate)

        # Graph learner (dynamic adjacency)
        self.graphGen = GraphLearner(self.hidR)

        # Graph Attention layers
        self.GAttnBlocks = nn.ModuleList([
            GraphAttnLayer(in_features=self.hidR, out_features=self.hidR)
            for _ in range(self.n)
        ])

        # Final projection
        self.output = nn.Linear(self.hidR * self.n + self.hidR, num_timesteps_output)
        self.init_weights()

    def init_weights(self):
        """
        Initializes weights of the model using Xavier uniform initialization.
        """
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                stdv = 1.0 / math.sqrt(p.size(0))
                p.data.uniform_(-stdv, stdv)

    def forward(self, X: torch.Tensor, adj: torch.Tensor, adjacency_type: str = 'hybrid') -> torch.Tensor:
        """
        Forward pass for EpiGNN.

        Parameters:
        -----------
        X : torch.Tensor
            Input tensor of shape (batch_size, T, m, F).
        adj : torch.Tensor
            Adjacency matrix of shape (batch_size, m, m).
        adjacency_type : str, optional
            Type of adjacency to use ('static', 'dynamic', 'hybrid'), by default 'hybrid'.

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (batch_size, T_out, m).
        """
        # RegionAwareConv
        X_reshaped = X.permute(0, 3, 1, 2)  # => (batch_size, F, T, m)
        temp_emb = self.backbone(X_reshaped)  # => (batch_size, m, hidR)

        # Q/K transformations to generate adjacency
        query = self.dropout_layer(self.WQ(temp_emb))  # (batch_size, m, hidA)
        key = self.dropout_layer(self.WK(temp_emb))    # (batch_size, m, hidA)

        attn = torch.bmm(query, key.transpose(1, 2))  # (batch_size, m, m)
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1, keepdim=True)  # (batch_size, m, 1)
        t_enc = self.dropout_layer(self.t_enc(attn))  # (batch_size, m, hidR)

        # Spatial encoding
        d = torch.sum(adj, dim=2, keepdim=True)  # (batch_size, m, 1)
        s_enc = self.dropout_layer(self.s_enc(d))  # (batch_size, m, hidR)

        # Feature Embedding
        feat_emb = temp_emb + t_enc + s_enc  # (batch_size, m, hidR)

        # Dynamic adjacency via GraphLearner
        learned_adj = self.graphGen(feat_emb)  # (batch_size, m, m)

        # Combine adjacency matrices
        if adjacency_type == 'static':
            combined_adj = adj
        elif adjacency_type == 'dynamic':
            combined_adj = learned_adj
        elif adjacency_type == 'hybrid':
            # Compute spatial adjacency
            d_mat = torch.sum(adj, dim=1, keepdim=True) * torch.sum(adj, dim=2, keepdim=True)  # (batch_size, m, m)
            d_mat = torch.sigmoid(self.d_gate.unsqueeze(0) * d_mat)  # Broadcasting d_gate to (1, m, m)
            spatial_adj = d_mat * adj  # (batch_size, m, m)
            combined_adj = torch.clamp(learned_adj + spatial_adj, 0, 1)
        else:
            raise ValueError("adjacency_type must be 'static', 'dynamic', or 'hybrid'.")

        # Apply Dropout
        feat_emb = self.dropout_layer(feat_emb)

        # Graph Attention Blocks
        node_state = feat_emb
        node_states_list = []
        for layer in self.GAttnBlocks:
            node_state = self.dropout_layer(layer(node_state, combined_adj))
            node_states_list.append(node_state)

        # Final concatenation
        node_cat = torch.cat(node_states_list, dim=-1)  # (batch_size, m, hidR*n)
        node_cat = torch.cat([node_cat, feat_emb], dim=-1)  # (batch_size, m, hidR*n + hidR)

        # Final Projection
        out = self.output(node_cat)  # (batch_size, m, T_out)
        return out.transpose(1, 2)   # (batch_size, T_out, m)

# ==============================================================================
# 8. Directory Setup and Visualization
# ==============================================================================
def setup_directories():
    """
    Creates necessary directories for models, figures, and results.
    """
    os.makedirs('figures', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('results/metrics', exist_ok=True)
    logging.info("Directories set up successfully.")

def visualize_adjacency(adj_matrix: torch.Tensor, regions: list, latitudes: list, longitudes: list, save_path: str):
    """
    Visualizes the geographic adjacency matrix as a graph.

    Parameters:
    -----------
    adj_matrix : torch.Tensor
        Adjacency matrix of shape (m, m).
    regions : list
        List of region names.
    latitudes : list
        List of latitudes corresponding to regions.
    longitudes : list
        List of longitudes corresponding to regions.
    save_path : str
        Path to save the adjacency graph plot.
    """
    adj_np = adj_matrix.cpu().numpy()
    G = nx.from_numpy_array(adj_np)
    mapping = {i: region for i, region in enumerate(regions)}
    G = nx.relabel_nodes(G, mapping)
    pos = {region: (longitudes[i], latitudes[i]) for i, region in enumerate(regions)}

    plt.figure(figsize=(12, 10))
    nx.draw_networkx(G, pos, with_labels=True,
                     node_size=1000, node_color='lightblue',
                     edge_color='gray', font_size=12,
                     font_weight='bold')
    plt.title('Geographic Adjacency Graph - Static Adjacency')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    logging.info(f"Geographic adjacency graph saved to {save_path}")

# ==============================================================================
# 9. Model Initialization
# ==============================================================================
def initialize_model():
    """
    Initializes the EpiGNN model along with optimizer, loss, and scheduler.

    Returns:
    --------
    model : nn.Module
        The initialized model.
    optimizer : torch.optim.Optimizer
        The optimizer.
    criterion : nn.Module
        The loss function.
    scheduler : torch.optim.lr_scheduler
        The learning rate scheduler.
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
        dropout=DROPOUT,
        device=device
    ).to(device)
    logging.info("Model initialized successfully.")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    logging.info("Optimizer, loss function, and scheduler initialized.")
    return model, optimizer, criterion, scheduler

# ==============================================================================
# 10. Training & Evaluation Function
# ==============================================================================
def run_experiment(adjacency_type: str = 'hybrid', experiment_id: int = 1, summary_metrics: list = None,
                  train_loader: DataLoader = None, val_loader: DataLoader = None, test_loader: DataLoader = None,
                  adj_static: torch.Tensor = None, scaled_dataset: NHSRegionDataset = None,
                  regions: list = None, data: pd.DataFrame = None, test_subset: Subset = None,
                  train_size: int = None, val_size: int = None):
    """
    Run a full training/validation/testing cycle for the EpiGNN model with a specified adjacency type.

    Parameters:
    -----------
    adjacency_type : str
        Type of adjacency to use ('static', 'dynamic', 'hybrid').
    experiment_id : int
        Identifier for the experiment.
    summary_metrics : list
        List to append metrics dictionaries.
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

            # Prepare adjacency input
            B_now = batch_X.size(0)
            if adjacency_type == 'static':
                adj_input = adj_static.unsqueeze(0).repeat(B_now, 1, 1)  # (B, m, m)
            elif adjacency_type == 'dynamic':
                adj_input = torch.zeros_like(adj_static).unsqueeze(0).repeat(B_now, 1, 1)  # (B, m, m)
            elif adjacency_type == 'hybrid':
                adj_input = adj_static.unsqueeze(0).repeat(B_now, 1, 1)  # (B, m, m)
            else:
                raise ValueError("adjacency_type must be 'static', 'dynamic', or 'hybrid'.")

            # Forward pass
            pred = model(batch_X, adj_input, adjacency_type=adjacency_type)  # (B, T_out, m)
            loss = criterion(pred, batch_Y)
            loss.backward()

            # Gradient clipping
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
                B_now = batch_X.size(0)
                if adjacency_type == 'static':
                    adj_input = adj_static.unsqueeze(0).repeat(B_now, 1, 1)
                elif adjacency_type == 'dynamic':
                    adj_input = torch.zeros_like(adj_static).unsqueeze(0).repeat(B_now, 1, 1)
                elif adjacency_type == 'hybrid':
                    adj_input = adj_static.unsqueeze(0).repeat(B_now, 1, 1)
                else:
                    adj_input = adj_static.unsqueeze(0).repeat(B_now, 1, 1)

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
        preds_2d = all_val_preds.reshape(-1, NUM_NODES)  # (B_total * T_out, m)
        actuals_2d = all_val_actuals.reshape(-1, NUM_NODES)  # (B_total * T_out, m)
        r2_vals = r2_score(actuals_2d, preds_2d, multioutput='raw_values')  # (m,)

        logging.info(f"Epoch {epoch}/{NUM_EPOCHS} - "
                     f"Train Loss: {avg_train_loss:.4f} | "
                     f"Val Loss: {avg_val_loss:.4f} | R² per node: {r2_vals}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_dir = f'models/experiment{experiment_id}'
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
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, len(train_losses) + 1), y=train_losses,
                 label='Train Loss', color='blue', marker='o')
    sns.lineplot(x=range(1, len(val_losses) + 1), y=val_losses,
                 label='Validation Loss', color='orange', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title(f'Training vs. Validation Loss - Experiment {experiment_id} ({adjacency_type.capitalize()})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    loss_plot_path = f'figures/training_validation_loss_experiment{experiment_id}_{adjacency_type}.png'
    plt.savefig(loss_plot_path, dpi=300)
    plt.show()
    logging.info(f"Loss curves saved to {loss_plot_path}")

    # Load best model for testing
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # Testing
    test_loss = 0.0
    all_preds = []
    all_actuals = []
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            B_now = batch_X.size(0)
            if adjacency_type == 'static':
                adj_input = adj_static.unsqueeze(0).repeat(B_now, 1, 1)
            elif adjacency_type == 'dynamic':
                adj_input = torch.zeros_like(adj_static).unsqueeze(0).repeat(B_now, 1, 1)
            elif adjacency_type == 'hybrid':
                adj_input = adj_static.unsqueeze(0).repeat(B_now, 1, 1)
            else:
                adj_input = adj_static.unsqueeze(0).repeat(B_now, 1, 1)

            pred = model(batch_X, adj_input, adjacency_type=adjacency_type)
            loss = criterion(pred, batch_Y)
            test_loss += loss.item()

            all_preds.append(pred.cpu())
            all_actuals.append(batch_Y.cpu())

    avg_test_loss = test_loss / len(test_loader)
    logging.info(f"Experiment {experiment_id} - Test Loss (MSE): {avg_test_loss:.4f}")

    # Reshape predictions and actuals
    all_preds = torch.cat(all_preds, dim=0).numpy()   # (B_test, T_out, m)
    all_actuals = torch.cat(all_actuals, dim=0).numpy()  # (B_test, T_out, m)

    # Inverse transform if scaler is used
    if scaled_dataset.scaler is not None:
        scale_covid = scaled_dataset.scaler.scale_[4]  # 'covidOccupiedMVBeds' index
        mean_covid  = scaled_dataset.scaler.mean_[4]
        all_preds_np = all_preds * scale_covid + mean_covid
        all_actuals_np = all_actuals * scale_covid + mean_covid
    else:
        all_preds_np = all_preds
        all_actuals_np = all_actuals

    # Flatten for metric computation
    preds_flat   = all_preds_np.reshape(-1, NUM_NODES)  # (B_test * T_out, m)
    actuals_flat = all_actuals_np.reshape(-1, NUM_NODES)  # (B_test * T_out, m)

    # Compute metrics per node
    mae_per_node = mean_absolute_error(actuals_flat, preds_flat, multioutput='raw_values')
    mse_per_node = mean_squared_error(actuals_flat, preds_flat, multioutput='raw_values')
    rmse_per_node = np.sqrt(mse_per_node)
    r2_per_node  = r2_score(actuals_flat, preds_flat, multioutput='raw_values')

    # Compute Pearson Correlation Coefficient per node
    pearson_per_node = []
    for i in range(NUM_NODES):
        if np.std(preds_flat[:, i]) == 0 or np.std(actuals_flat[:, i]) == 0:
            pearson_cc = 0
        else:
            pearson_cc, _ = pearsonr(preds_flat[:, i], actuals_flat[:, i])
        pearson_per_node.append(pearson_cc)

    # Organize metrics into a DataFrame
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

    metrics_df = pd.DataFrame(metrics_dict)
    metrics_csv_path = f'results/metrics/metrics_experiment{experiment_id}_{adjacency_type}.csv'
    metrics_df.to_csv(metrics_csv_path, index=False)
    logging.info(f"Metrics saved to {metrics_csv_path}")

    # ==============================================================================
    # 11. Enhanced Visualization
    # ==============================================================================
    unique_dates = data['date'].sort_values().unique()
    forecast_dates = []
    num_test_samples = len(test_subset)

    for i in range(num_test_samples):
        pred_start_idx = train_size + val_size + i + NUM_TIMESTEPS_INPUT
        pred_end_idx   = pred_start_idx + NUM_TIMESTEPS_OUTPUT
        if pred_end_idx > len(unique_dates):
            pred_end_idx = len(unique_dates)
        sample_dates = unique_dates[pred_start_idx : pred_end_idx]
        if len(sample_dates) < NUM_TIMESTEPS_OUTPUT:
            last_date = unique_dates[-1]
            sample_dates = np.append(sample_dates, [last_date] * (NUM_TIMESTEPS_OUTPUT - len(sample_dates)))
        forecast_dates.extend(sample_dates)

    # Ensure forecast_dates length matches all_preds_np
    expected_length = all_preds_np.shape[0] * NUM_TIMESTEPS_OUTPUT
    if len(forecast_dates) < expected_length:
        # Pad with the last available date
        last_date = unique_dates[-1]
        forecast_dates.extend([last_date] * (expected_length - len(forecast_dates)))
    elif len(forecast_dates) > expected_length:
        forecast_dates = forecast_dates[:expected_length]

    # Build DataFrames for plotting
    preds_df = pd.DataFrame(all_preds_np.reshape(-1, NUM_NODES), columns=regions)
    preds_df['Date'] = forecast_dates

    actuals_df = pd.DataFrame(all_actuals_np.reshape(-1, NUM_NODES), columns=regions)
    actuals_df['Date'] = forecast_dates

    # Aggregate predictions and actuals by date (mean)
    agg_preds_df  = preds_df.groupby('Date').mean().reset_index()
    agg_actuals_df= actuals_df.groupby('Date').first().reset_index()

    merged_df = pd.merge(agg_preds_df, agg_actuals_df, on='Date', suffixes=('_Predicted', '_Actual'))
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])

    def plot_time_series(region: str, df: pd.DataFrame, adjacency_type: str, experiment_id: int):
        """
        Plots the actual vs predicted time series for a specific region.

        Parameters:
        -----------
        region : str
            The name of the region.
        df : pd.DataFrame
            DataFrame containing the aggregated actual and predicted values.
        adjacency_type : str
            Type of adjacency used.
        experiment_id : int
            Identifier for the experiment.
        """
        plt.figure(figsize=(14, 7))
        plt.plot(df['Date'], df[f'{region}_Actual'], label='Actual', color='blue')
        plt.plot(df['Date'], df[f'{region}_Predicted'], label='Predicted', color='red')
        plt.title(f'Actual vs Predicted COVID Occupied MV Beds - {region} ({adjacency_type.capitalize()})')
        plt.xlabel('Date')
        plt.ylabel('COVID Occupied MV Beds')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = f'figures/time_series_{region}_experiment{experiment_id}_{adjacency_type}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
        logging.info(f"Time-series plot for {region} saved to {plot_path}")

    for region in regions:
        plot_time_series(region, merged_df, adjacency_type, experiment_id)

    # Additional Visualizations:

    def plot_error_distribution(region: str, df: pd.DataFrame, adjacency_type: str, experiment_id: int):
        """
        Plots the distribution of prediction errors for a specific region.

        Parameters:
        -----------
        region : str
            The name of the region.
        df : pd.DataFrame
            DataFrame containing the aggregated actual and predicted values.
        adjacency_type : str
            Type of adjacency used.
        experiment_id : int
            Identifier for the experiment.
        """
        errors = df[f'{region}_Predicted'] - df[f'{region}_Actual']
        plt.figure(figsize=(10, 6))
        sns.histplot(errors, bins=30, kde=True, color='purple')
        plt.title(f'Prediction Error Distribution for {region} ({adjacency_type.capitalize()} Adjacency)')
        plt.xlabel('Prediction Error (Predicted - Actual)')
        plt.ylabel('Frequency')

        mean_error   = errors.mean()
        median_error = errors.median()
        plt.axvline(mean_error,   color='red',   linestyle='dashed', linewidth=1, label=f'Mean: {mean_error:.2f}')
        plt.axvline(median_error, color='green', linestyle='dotted', linewidth=1, label=f'Median: {median_error:.2f}')

        plt.legend()
        plt.tight_layout()
        plot_path = f'figures/error_distribution_experiment{experiment_id}_{adjacency_type}_{region.replace(" ", "_")}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
        logging.info(f"Error distribution plot for {region} saved to {plot_path}")

    for region in regions:
        plot_error_distribution(region, merged_df, adjacency_type, experiment_id)

    def plot_cumulative_error(region: str, df: pd.DataFrame, adjacency_type: str, experiment_id: int):
        """
        Plots the cumulative prediction error over time for a specific region.

        Parameters:
        -----------
        region : str
            The name of the region.
        df : pd.DataFrame
            DataFrame containing the aggregated actual and predicted values.
        adjacency_type : str
            Type of adjacency used.
        experiment_id : int
            Identifier for the experiment.
        """
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
        logging.info(f"Cumulative error plot for {region} saved to {plot_path}")

    for region in regions:
        plot_cumulative_error(region, merged_df, adjacency_type, experiment_id)

    def plot_error_boxplot(df: pd.DataFrame, adjacency_type: str, experiment_id: int):
        """
        Plots a boxplot of prediction errors for each region.

        Parameters:
        -----------
        df : pd.DataFrame
            Merged DataFrame containing predicted and actual values.
        adjacency_type : str
            Type of adjacency used.
        experiment_id : int
            Identifier for the experiment.
        """
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
        logging.info(f"Boxplot of prediction errors saved to {plot_path}")

    plot_error_boxplot(merged_df, adjacency_type, experiment_id)

    def plot_scatter_actual_vs_predicted(region: str, df: pd.DataFrame, adjacency_type: str, experiment_id: int):
        """
        Plots a scatter plot of actual vs. predicted values for a specific region.

        Parameters:
        -----------
        region : str
            The name of the region.
        df : pd.DataFrame
            Merged DataFrame containing predicted and actual values.
        adjacency_type : str
            Type of adjacency used.
        experiment_id : int
            Identifier for the experiment.
        """
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
        logging.info(f"Scatter plot for {region} saved to {plot_path}")

    for region in regions:
        plot_scatter_actual_vs_predicted(region, merged_df, adjacency_type, experiment_id)

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
        logging.info(f"Heatmap of prediction errors for {region} saved to {plot_path}")

    # Scatter Plot: Actual vs Predicted (Overall)
    def plot_scatter_actual_vs_predicted_overall(df: pd.DataFrame, adjacency_type: str, experiment_id: int):
        """
        Plots a scatter plot of actual vs. predicted values averaged across regions.

        Parameters:
        -----------
        df : pd.DataFrame
            Merged DataFrame containing predicted and actual values.
        adjacency_type : str
            Type of adjacency used.
        experiment_id : int
            Identifier for the experiment.
        """
        # Compute average across regions
        df_overall = df.copy()
        df_overall['covidOccupiedMVBeds_Actual_Avg'] = df_overall[[f'{region}_Actual' for region in regions]].mean(axis=1)
        df_overall['covidOccupiedMVBeds_Predicted_Avg'] = df_overall[[f'{region}_Predicted' for region in regions]].mean(axis=1)

        plt.figure(figsize=(7, 7))
        sns.scatterplot(x='covidOccupiedMVBeds_Actual_Avg',
                        y='covidOccupiedMVBeds_Predicted_Avg',
                        data=df_overall, color='teal', alpha=0.6)
        sns.regplot(x='covidOccupiedMVBeds_Actual_Avg',
                    y='covidOccupiedMVBeds_Predicted_Avg',
                    data=df_overall,
                    scatter=False, color='red', label='Regression Line')
        plt.title(f'Overall Actual vs Predicted (Avg) ({adjacency_type.capitalize()})')
        plt.xlabel('Actual COVID Occupied MV Beds')
        plt.ylabel('Predicted COVID Occupied MV Beds')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = f'figures/scatter_actual_vs_predicted_overall_experiment{experiment_id}_{adjacency_type}.png'
        plt.savefig(plot_path, dpi=300)
        plt.show()
        logging.info(f"Overall scatter plot saved to {plot_path}")

    plot_scatter_actual_vs_predicted_overall(merged_df, adjacency_type, experiment_id)

    # ==============================================================================
    # 12. Save Final Model
    # ==============================================================================
    final_model_path = f'models/experiment{experiment_id}/{adjacency_type}_final_model.pth'
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved as '{final_model_path}'.")

    logging.info(f"Experiment {experiment_id} workflow complete.\n")

# ==============================================================================
# 11. Main Execution
# ==============================================================================
def main():
    """
    Main function to execute experiments.
    """
    # Setup directories
    setup_directories()

    # Load and preprocess data
    csv_path = "data/merged_nhs_covid_data.csv"
    if not os.path.exists(csv_path):
        logging.error(f"CSV file not found at {csv_path}")
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    data = pd.read_csv(csv_path, parse_dates=['date'])
    data = load_and_correct_data(data, REFERENCE_COORDINATES)

    # Build the dataset
    initial_dataset = NHSRegionDataset(
        data,
        num_timesteps_input=NUM_TIMESTEPS_INPUT,
        num_timesteps_output=NUM_TIMESTEPS_OUTPUT,
        scaler=None
    )
    logging.info(f"Total samples in initial dataset: {len(initial_dataset)}")

    # Fixed chronological split: 70% train, 15% validation, 15% test
    total_len = len(initial_dataset)
    train_size = int(0.7 * total_len)
    val_size = int(0.15 * total_len)
    test_size = total_len - train_size - val_size

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_len))

    # Fit scaler on training data
    scaler = StandardScaler()
    train_features = []
    for i in train_indices:
        X, _ = initial_dataset[i]
        train_features.append(X.numpy())

    train_features = np.concatenate(train_features, axis=0).reshape(-1, NUM_FEATURES)
    scaler.fit(train_features)
    logging.info("Scaler fitted on training data.")

    # Create scaled dataset
    scaled_dataset = NHSRegionDataset(
        data,
        num_timesteps_input=NUM_TIMESTEPS_INPUT,
        num_timesteps_output=NUM_TIMESTEPS_OUTPUT,
        scaler=scaler
    )
    logging.info(f"Total samples in scaled dataset: {len(scaled_dataset)}")

    # Create subsets
    train_subset = Subset(scaled_dataset, train_indices)
    val_subset   = Subset(scaled_dataset, val_indices)
    test_subset  = Subset(scaled_dataset, test_indices)

    logging.info(f"Training samples:   {len(train_subset)}")
    logging.info(f"Validation samples: {len(val_subset)}")
    logging.info(f"Test samples:       {len(test_subset)}")

    # Create DataLoaders
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE,
                              shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_subset,  batch_size=BATCH_SIZE,
                              shuffle=False, drop_last=False)

    # Compute adjacency matrix
    regions = scaled_dataset.regions.tolist()
    latitudes  = [data[data['areaName'] == region]['latitude'].iloc[0] for region in regions]
    longitudes = [data[data['areaName'] == region]['longitude'].iloc[0] for region in regions]

    adj_static = compute_geographic_adjacency(regions, latitudes, longitudes, THRESHOLD_DISTANCE).to(device)
    logging.info("Static Adjacency Matrix:")
    logging.info(adj_static.cpu().numpy())

    # Visualize adjacency
    visualize_adjacency(adj_static, regions, latitudes, longitudes, 
                       save_path='figures/geographic_adjacency_graph_static.png')

    # Define experiments
    experiments = [
        {'adjacency_type': 'static',  'experiment_id': 1},
        {'adjacency_type': 'dynamic', 'experiment_id': 2},
        {'adjacency_type': 'hybrid',  'experiment_id': 3},
    ]
    summary_metrics = []

    # Execute each experiment
    for exp in experiments:
        run_experiment(
            adjacency_type=exp['adjacency_type'],
            experiment_id=exp['experiment_id'],
            summary_metrics=summary_metrics,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            adj_static=adj_static,  # Pass adj_static here
            scaled_dataset=scaled_dataset,  # Pass scaled_dataset here
            regions=regions,  # Pass regions here
            data=data,  # Pass data here
            test_subset=test_subset,  # Pass test_subset here
            train_size=train_size,  # Pass train_size here
            val_size=val_size       # Pass val_size here
        )

    # Summarize results
    summary_df = pd.DataFrame(summary_metrics)
    summary_csv = 'results/metrics/summary_metrics.csv'
    summary_df.to_csv(summary_csv, index=False)
    logging.info(f"Summary metrics saved to {summary_csv}")

    # Pivot to show average metrics per experiment
    summary_pivot = summary_df.groupby(['Experiment_ID', 'Adjacency_Type']).agg({
        'MAE': 'mean',
        'MSE': 'mean',
        'RMSE': 'mean',
        'R2_Score': 'mean',
        'Pearson_CC': 'mean'
    }).reset_index()

    logging.info(f"Summary of All Experiments:\n{summary_pivot}")

    summary_pivot_path = 'results/metrics/summary_metrics_pivot.csv'
    summary_pivot.to_csv(summary_pivot_path, index=False)
    logging.info(f"Summary pivot table saved to {summary_pivot_path}")

    # Optional comparison plot for summary metrics
    def plot_summary_metrics(summary_pivot: pd.DataFrame):
        """
        Plots average metrics across different adjacency types.

        Parameters:
        -----------
        summary_pivot : pd.DataFrame
            Pivoted summary metrics DataFrame.
        """
        metrics_list = ['MAE','MSE','RMSE','R2_Score','Pearson_CC']
        plt.figure(figsize=(20, 12))
        
        # Determine grid size based on number of metrics
        num_metrics = len(metrics_list)
        cols = 3
        rows = math.ceil(num_metrics / cols)
        
        for i, metric in enumerate(metrics_list):
            plt.subplot(rows, cols, i+1)
            
            sns.barplot(x='Adjacency_Type', y=metric, data=summary_pivot, palette='Set2')
            
            plt.title(f'Average {metric}')
            plt.ylabel(metric)
            plt.xlabel('Adjacency Type')
            
            # Dynamic y-axis limits
            min_val = summary_pivot[metric].min()
            max_val = summary_pivot[metric].max()
            if metric in ['R2_Score', 'Pearson_CC']:
                plt.ylim(min(-1, min_val * 1.2), max(1, max_val * 1.2))
            else:
                plt.ylim(0, max(max_val, 1) * 1.2)
            
            # Text annotations with safe y positions
            for idx, row in summary_pivot.iterrows():
                x_pos = list(summary_pivot['Adjacency_Type']).index(row['Adjacency_Type'])
                y_pos = row[metric] + 0.01 * max_val
                plt.text(x_pos, y_pos, f"{row[metric]:.4f}",
                        ha='center', va='bottom', fontsize=10)
            
        plt.tight_layout()
        summary_plot = 'figures/summary_metrics_comparison.png'
        plt.savefig(summary_plot, dpi=300)
        plt.show()
        logging.info(f"Summary metrics comparison plot saved to {summary_plot}")

    plot_summary_metrics(summary_pivot)

if __name__ == "__main__":
    main()
