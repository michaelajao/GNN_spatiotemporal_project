#!/usr/bin/env python3
"""
EpiGNN (Daily Data, Static/Dynamic/Hybrid Experiments)
=====================================================

Uses daily COVID-19 data in a sliding-window approach, trains EpiGNN for multiple
adjacency experiment types: static, dynamic, and hybrid. Generates high-resolution figures
suitable for research paper usage.

Author: [Your Name]
Year: 2025
"""

import os
import math
import random
import logging
import warnings
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from tqdm.auto import tqdm
from tqdm import TqdmWarning

# ------------------------------------------------------------------------------
# Set Default Style for Publication-Quality Plots
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

# ------------------------------------------------------------------------------
# Suppress Specific Warnings
# ------------------------------------------------------------------------------
warnings.filterwarnings("ignore", category=TqdmWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

sns.set(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

# ------------------------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------------------------
def setup_logging(project_root):
    log_file = os.path.join(project_root, "experiment.log")
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
    )

# ------------------------------------------------------------------------------
# Global Hyperparameters & Settings
# ------------------------------------------------------------------------------
RANDOM_SEED = 123
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 32  # Updated from 4 to 32
NUM_EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 20
LEARNING_RATE = 1e-3
THRESHOLD_DISTANCE = 300  # for static adjacency
NUM_TIMESTEPS_INPUT = 20  # look-back

NUM_NODES = 7
NUM_FEATURES = 5  # new_confirmed, new_deceased, newAdmissions, hospitalCases, covidOccupiedMVBeds

REFERENCE_COORDINATES = {
    "East of England": (52.1766, 0.425889),
    "Midlands": (52.7269, -1.458210),
    "London": (51.4923, -0.308660),
    "South East": (51.4341, -0.969570),
    "South West": (50.8112, -3.633430),
    "North West": (53.8981, -2.657550),
    "North East and Yorkshire": (54.5378, -2.180390),
}

# ------------------------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------------------------
def seed_everything(seed=RANDOM_SEED):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_and_correct_data_daily(data: pd.DataFrame, ref_coords: dict):
    """
    1) Assign lat/long from reference coords.
    2) Sort daily data by (areaName, date).
    3) Fill NAs with 0, do not perform weekly aggregation.
    """
    for region, coords in ref_coords.items():
        data.loc[data["areaName"] == region, ["latitude", "longitude"]] = coords

    # Fill NAs for daily data
    daily_feats = ["new_confirmed", "new_deceased", "newAdmissions", "hospitalCases", "covidOccupiedMVBeds"]
    data[daily_feats] = data[daily_feats].fillna(0)

    data.sort_values(["areaName", "date"], inplace=True)
    logging.info("Data loaded (daily) and coordinates assigned. Sorted by areaName and date.")
    return data

# ------------------------------------------------------------------------------
# Dataset Class
# ------------------------------------------------------------------------------
class DailyDataset(Dataset):
    """
    Takes daily data in "long" form. Builds a sliding window for each region.
    - X: shape (num_timesteps_input, num_nodes, num_features)
    - Y: shape (num_timesteps_output, num_nodes)
    """
    def __init__(self, data: pd.DataFrame, num_timesteps_input=20, num_timesteps_output=7, scaler=None):
        super().__init__()
        self.data = data.copy()
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output

        # Region indexing
        self.regions = self.data["areaName"].unique()
        self.num_nodes = len(self.regions)
        self.region_to_idx = {r: i for i, r in enumerate(self.regions)}
        self.data["region_idx"] = self.data["areaName"].map(self.region_to_idx)

        # Relevant features
        self.features = ["new_confirmed", "new_deceased", "newAdmissions", "hospitalCases", "covidOccupiedMVBeds"]

        # Pivot: index=date, columns=region_idx, values=features => shape (#days, #nodes, #features)
        pivoted = self.data.pivot(index="date", columns="region_idx", values=self.features)
        pivoted.ffill(inplace=True)  # Forward fill
        pivoted.fillna(0, inplace=True)

        self.num_days = pivoted.shape[0]
        self.num_features = len(self.features)
        self.feature_array = pivoted.values.reshape(self.num_days, self.num_nodes, self.num_features)

        self.scaler = scaler
        if self.scaler is not None:
            arr_2d = self.feature_array.reshape(-1, self.num_features)
            arr_2d = self.scaler.fit_transform(arr_2d)
            self.feature_array = arr_2d.reshape(self.num_days, self.num_nodes, self.num_features)

    def __len__(self):
        return self.num_days - self.num_timesteps_input - self.num_timesteps_output + 1

    def __getitem__(self, idx):
        X = self.feature_array[idx : idx + self.num_timesteps_input]  # (T_in, num_nodes, num_feats)
        Y = self.feature_array[idx + self.num_timesteps_input : idx + self.num_timesteps_input + self.num_timesteps_output, :, 4]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# ------------------------------------------------------------------------------
# GAT, GraphLearner, EpiGNN, etc.
# ------------------------------------------------------------------------------
class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=4, dropout=0.2, alpha=0.2):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.a = nn.Parameter(torch.Tensor(num_heads, 2 * out_features))
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a, gain=1.414)

    def forward(self, h, adj):
        bs, n, _ = h.size()
        Wh = self.W(h).view(bs, n, self.num_heads, self.out_features)
        Wh = Wh.permute(0, 2, 1, 3)  # (bs, heads, n, outfeat)

        # Prepare attention
        a_input = torch.cat(
            [
                Wh.unsqueeze(3).repeat(1, 1, 1, n, 1),
                Wh.unsqueeze(2).repeat(1, 1, n, 1, 1)
            ],
            dim=-1
        )  # (bs, heads, n, n, 2*outfeat)

        a_expanded = self.a.unsqueeze(0).unsqueeze(2).unsqueeze(3)  # (1, heads, 1, 1, 2*outfeat)
        e = self.leakyrelu((a_input * a_expanded).sum(dim=-1))  # (bs, heads, n, n)

        e = e.masked_fill(adj.unsqueeze(1) == 0, float("-inf"))
        attn = torch.softmax(e, dim=-1)
        attn = self.dropout(attn)

        h_prime = torch.matmul(attn, Wh)  # (bs, heads, n, outfeat)
        h_prime = h_prime.permute(0, 2, 1, 3).contiguous()  # (bs, n, heads, outfeat)
        h_prime = h_prime.view(bs, n, self.num_heads * self.out_features)
        return h_prime

class GraphLearner(nn.Module):
    def __init__(self, hidden_dim, tanhalpha=1):
        super().__init__()
        self.hid = hidden_dim
        self.linear1 = nn.Linear(self.hid, self.hid)
        self.linear2 = nn.Linear(self.hid, self.hid)
        self.alpha = tanhalpha

    def forward(self, emb):
        """
        emb: (bs, n, hidR)
        """
        bs, n, _ = emb.size()
        n1 = torch.tanh(self.alpha * self.linear1(emb))
        n2 = torch.tanh(self.alpha * self.linear2(emb))
        adj = (torch.bmm(n1, n2.transpose(1, 2)) - torch.bmm(n2, n1.transpose(1, 2))) * self.alpha
        adj = torch.relu(torch.tanh(adj))

        eye = torch.eye(n, device=adj.device).unsqueeze(0).repeat(bs, 1, 1)
        adj = adj + eye
        adj = torch.clamp(adj, min=1e-6, max=1.0)
        return adj

class ConvBranch(nn.Module):
    def __init__(self, m, in_channels, out_channels, kernel_size, dilation_factor=2, hidP=1, isPool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), dilation=(dilation_factor, 1))
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.isPool = isPool
        if self.isPool and hidP is not None:
            self.pooling = nn.AdaptiveMaxPool2d((hidP, m))
        self.activate = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        if self.isPool and hasattr(self, "pooling"):
            x = self.pooling(x)
        bs = x.size(0)
        x = x.view(bs, -1, x.size(-1))  # (bs, out_channels*hidP, m)
        return self.activate(x)

class RegionAwareConv(nn.Module):
    def __init__(self, nfeat, P, m, k, hidP, dilation_factor=2):
        super().__init__()
        self.conv_l1 = ConvBranch(m, nfeat, k, kernel_size=3, dilation_factor=1, hidP=hidP)
        self.conv_l2 = ConvBranch(m, nfeat, k, kernel_size=5, dilation_factor=1, hidP=hidP)
        self.conv_p1 = ConvBranch(m, nfeat, k, kernel_size=3, dilation_factor=dilation_factor, hidP=hidP)
        self.conv_p2 = ConvBranch(m, nfeat, k, kernel_size=5, dilation_factor=dilation_factor, hidP=hidP)
        self.conv_g  = ConvBranch(m, nfeat, k, kernel_size=P, dilation_factor=1, hidP=None, isPool=False)
        self.activate = nn.Tanh()

    def forward(self, x):
        # x: (bs, feats, T, m)
        xl1 = self.conv_l1(x)
        xl2 = self.conv_l2(x)
        x_local = torch.cat([xl1, xl2], dim=1)

        xp1 = self.conv_p1(x)
        xp2 = self.conv_p2(x)
        x_period = torch.cat([xp1, xp2], dim=1)

        xg = self.conv_g(x)

        xcat = torch.cat([x_local, x_period, xg], dim=1)
        return self.activate(xcat).permute(0, 2, 1)  # (bs, m, something)

class EpiGNN(nn.Module):
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
                 num_heads=4,
                 dropout=0.5,
                 device="cpu"):
        super().__init__()
        self.device = device
        self.m = num_nodes
        self.w = num_timesteps_input
        self.hidR = hidR
        self.hidA = hidA
        self.hidP = hidP
        self.k = k
        self.n = n_layer
        self.dropout_layer = nn.Dropout(dropout)

        self.backbone = RegionAwareConv(nfeat=num_features, P=self.w, m=self.m, k=self.k, hidP=self.hidP)
        self.WQ = nn.Linear(self.hidR, self.hidA)
        self.WK = nn.Linear(self.hidR, self.hidA)
        self.t_enc = nn.Linear(1, self.hidR)
        self.s_enc = nn.Linear(1, self.hidR)

        self.d_gate = nn.Parameter(torch.FloatTensor(self.m, self.m), requires_grad=True)
        nn.init.xavier_uniform_(self.d_gate)

        self.graphGen = GraphLearner(self.hidR)

        self.GATLayers = nn.ModuleList([
            MultiHeadGATLayer(self.hidR, self.hidR // num_heads, num_heads=num_heads, dropout=dropout)
            for _ in range(self.n)
        ])

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

    def forward(self, X, adj, adjacency_type="static"):
        """
        X: (bs, T_in, m, feats)
        adj: (bs, m, m)
        adjacency_type: static, dynamic, or hybrid
        """
        # RegionAwareConv expects shape: (bs, feats, T, m)
        X_reshaped = X.permute(0, 3, 1, 2)
        temp_emb = self.backbone(X_reshaped)  # (bs, m, hidR)

        query = self.dropout_layer(self.WQ(temp_emb))  # (bs, m, hidA)
        key   = self.dropout_layer(self.WK(temp_emb))  # (bs, m, hidA)

        attn = torch.bmm(query, key.transpose(1, 2))
        attn = F.normalize(attn, dim=-1, p=2, eps=1e-12)
        attn = torch.sum(attn, dim=-1, keepdim=True)  # (bs, m, 1)
        t_enc = self.dropout_layer(self.t_enc(attn))

        d = torch.sum(adj, dim=2).unsqueeze(2)  # (bs, m, 1)
        s_enc = self.dropout_layer(self.s_enc(d))

        feat_emb = temp_emb + t_enc + s_enc  # (bs, m, hidR)
        learned_adj = self.graphGen(feat_emb)  # (bs, m, m)

        if adjacency_type == "static":
            combined_adj = adj
        elif adjacency_type == "dynamic":
            combined_adj = learned_adj
        elif adjacency_type == "hybrid":
            combined_adj = (learned_adj + adj)
        else:
            raise ValueError("Invalid adjacency_type: static|dynamic|hybrid")

        laplace_adj = getLaplaceMat(X.size(0), self.m, combined_adj)

        node_state = feat_emb
        gat_outputs = []
        for gat in self.GATLayers:
            node_state = gat(node_state, laplace_adj)
            node_state = F.elu(node_state)
            node_state = self.dropout_layer(node_state)
            gat_outputs.append(node_state)

        gat_cat = torch.cat(gat_outputs, dim=-1)
        node_state_all = torch.cat([gat_cat, feat_emb], dim=-1)
        res = self.output(node_state_all)  # (bs, m, T_out)
        return res.transpose(1, 2)         # => (bs, T_out, m)

# ------------------------------------------------------------------------------
# Utility: Adjacency, Laplacian, etc.
# ------------------------------------------------------------------------------
def haversine_distance(u, v):
    from math import radians, sin, cos, asin, sqrt
    lat1, lon1 = map(radians, [u[0], u[1]])
    lat2, lon2 = map(radians, [v[0], v[1]])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat / 2) ** 2) + cos(lat1) * cos(lat2) * (sin(dlon / 2) ** 2)
    c = 2 * asin(sqrt(a))
    return c * 6371  # Radius of Earth in kilometers

from scipy.spatial.distance import pdist, squareform

def compute_geographic_adjacency(regions, latitudes, longitudes, threshold=300):
    coords = np.column_stack((latitudes, longitudes))

    def custom_dist(u, v):
        return haversine_distance(u, v)

    dist_mat = squareform(pdist(coords, metric=custom_dist))
    adj_mat = (dist_mat <= threshold).astype(np.float32)
    np.fill_diagonal(adj_mat, 1.0)
    return torch.tensor(adj_mat, dtype=torch.float32)

def getLaplaceMat(bs, m, adj):
    i_mat = torch.eye(m, device=adj.device).unsqueeze(0).expand(bs, m, m)
    adj_bin = (adj > 0).float()
    deg = torch.sum(adj_bin, dim=2)
    deg_inv = 1.0 / (deg + 1e-12)
    deg_inv_mat = i_mat * deg_inv.unsqueeze(2)
    laplace = torch.bmm(deg_inv_mat, adj_bin)
    return laplace

# ------------------------------------------------------------------------------
# Initialize Model
# ------------------------------------------------------------------------------
def initialize_model(num_timesteps_output):
    model = EpiGNN(
        num_nodes=NUM_NODES,
        num_features=NUM_FEATURES,
        num_timesteps_input=NUM_TIMESTEPS_INPUT,
        num_timesteps_output=num_timesteps_output,
        k=8,
        hidA=32,
        hidR=40,
        hidP=1,
        n_layer=3,
        num_heads=4,
        dropout=0.5,
        device=device
    ).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.5, patience=3, verbose=True)
    logging.info("Model, optimizer, loss function, and scheduler initialized successfully.")
    return model, optimizer, criterion, scheduler

# ------------------------------------------------------------------------------
# Run Single Experiment (Given adjacency_type)
# ------------------------------------------------------------------------------
def run_experiment(horizon,
                   adjacency_type,  # "static"|"dynamic"|"hybrid"
                   experiment_id,
                   summary_metrics,
                   train_loader,
                   val_loader,
                   test_loader,
                   adj_static,
                   scaled_dataset,
                   regions,
                   data,
                   test_subset,
                   train_size,
                   val_size,
                   project_root):
    logging.info(f"\n--- Starting Experiment {experiment_id} with adjacency={adjacency_type}, horizon={horizon} ---")

    model, optimizer, criterion, scheduler = initialize_model(num_timesteps_output=horizon)

    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        epoch_train_loss = 0.0

        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()

            bs_cur = batch_X.size(0)
            # Adjacency
            if adjacency_type == "static":
                adj_input = adj_static.unsqueeze(0).repeat(bs_cur, 1, 1)
            elif adjacency_type == "dynamic":
                # Dummy adjacency; model's GraphLearner handles it
                adj_input = torch.eye(NUM_NODES, device=device).unsqueeze(0).repeat(bs_cur, 1, 1)
            elif adjacency_type == "hybrid":
                # Sum of static adjacency and learned adjacency inside the model
                adj_input = adj_static.unsqueeze(0).repeat(bs_cur, 1, 1)
            else:
                raise ValueError("Invalid adjacency type")

            pred = model(batch_X, adj_input, adjacency_type=adjacency_type)
            loss = criterion(pred, batch_Y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = (epoch_train_loss / len(train_loader)) if len(train_loader) > 0 else 0.0
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0.0
        all_val_preds = []
        all_val_actuals = []

        with torch.no_grad():
            for batch_Xv, batch_Yv in val_loader:
                batch_Xv, batch_Yv = batch_Xv.to(device), batch_Yv.to(device)
                bs_cur = batch_Xv.size(0)
                # Adjacency
                if adjacency_type == "static":
                    adj_input = adj_static.unsqueeze(0).repeat(bs_cur, 1, 1)
                elif adjacency_type == "dynamic":
                    adj_input = torch.eye(NUM_NODES, device=device).unsqueeze(0).repeat(bs_cur, 1, 1)
                elif adjacency_type == "hybrid":
                    adj_input = adj_static.unsqueeze(0).repeat(bs_cur, 1, 1)
                else:
                    raise ValueError("Invalid adjacency type")

                predv = model(batch_Xv, adj_input, adjacency_type=adjacency_type)
                vloss = criterion(predv, batch_Yv)
                epoch_val_loss += vloss.item()

                all_val_preds.append(predv.cpu().numpy())
                all_val_actuals.append(batch_Yv.cpu().numpy())

        avg_val_loss = (epoch_val_loss / len(val_loader)) if len(val_loader) > 0 else 0.0
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        # R² Calculation
        if all_val_preds and all_val_actuals:
            val_preds_arr = np.concatenate(all_val_preds, axis=0)  # (N, horizon, m)
            val_acts_arr  = np.concatenate(all_val_actuals, axis=0)
            preds_2d = val_preds_arr.reshape(-1, scaled_dataset.num_nodes)
            acts_2d  = val_acts_arr.reshape(-1, scaled_dataset.num_nodes)

            r2_vals = []
            for nd in range(scaled_dataset.num_nodes):
                if np.isnan(preds_2d[:, nd]).any() or np.isnan(acts_2d[:, nd]).any():
                    r2_vals.append(float("nan"))
                else:
                    r2_vals.append(r2_score(acts_2d[:, nd], preds_2d[:, nd]))
        else:
            r2_vals = [float("nan")] * scaled_dataset.num_nodes

        logging.info(f"Epoch {epoch}/{NUM_EPOCHS} - adjacency={adjacency_type} - "
                     f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
                     f"| R² per node: {r2_vals}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_dir = os.path.join(project_root, "models", f"experiment{experiment_id}_{adjacency_type}_h{horizon}")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"[BEST] Saved model -> {ckpt_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logging.info("Early stopping triggered.")
                break

    # Plot Train vs Val Loss
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=range(1, len(train_losses)+1), y=train_losses, label="Train Loss")
    sns.lineplot(x=range(1, len(val_losses)+1), y=val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Exp {experiment_id}, adjacency={adjacency_type}, horizon={horizon}")
    plt.grid(True)
    plt.legend()
    fig_dir = os.path.join(project_root, "figures", f"experiment{experiment_id}_{adjacency_type}_h{horizon}", "training_validation_loss")
    os.makedirs(fig_dir, exist_ok=True)
    fig_path = os.path.join(fig_dir, f"train_val_loss_experiment{experiment_id}_{adjacency_type}_h{horizon}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Loss curves saved -> {fig_path}")

    # Testing
    ckpt_path = os.path.join(project_root, "models", f"experiment{experiment_id}_{adjacency_type}_h{horizon}", "best_model.pth")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    test_loss = 0.0
    test_preds = []
    test_acts  = []

    with torch.no_grad():
        for batch_Xt, batch_Yt in test_loader:
            batch_Xt, batch_Yt = batch_Xt.to(device), batch_Yt.to(device)
            bs_cur = batch_Xt.size(0)

            if adjacency_type == "static":
                adj_input = adj_static.unsqueeze(0).repeat(bs_cur, 1, 1)
            elif adjacency_type == "dynamic":
                adj_input = torch.eye(NUM_NODES, device=device).unsqueeze(0).repeat(bs_cur, 1, 1)
            elif adjacency_type == "hybrid":
                adj_input = adj_static.unsqueeze(0).repeat(bs_cur, 1, 1)

            predt = model(batch_Xt, adj_input, adjacency_type=adjacency_type)
            tloss = criterion(predt, batch_Yt)
            test_loss += tloss.item()

            test_preds.append(predt.cpu().numpy())
            test_acts.append(batch_Yt.cpu().numpy())

    avg_test_loss = (test_loss / len(test_loader)) if len(test_loader) > 0 else float("nan")
    logging.info(f"Experiment {experiment_id}, adjacency={adjacency_type}, horizon={horizon} => Test MSE: {avg_test_loss:.4f}")

    # Combine Predictions
    if test_preds and test_acts:
        preds_arr = np.concatenate(test_preds, axis=0)  # (N, horizon, m)
        acts_arr  = np.concatenate(test_acts,  axis=0)
    else:
        preds_arr = np.array([])
        acts_arr  = np.array([])

    # Inverse Transform
    if preds_arr.size > 0 and scaled_dataset.scaler is not None:
        sc_covid = scaled_dataset.scaler.scale_[4]
        mn_covid = scaled_dataset.scaler.mean_[4]
        preds_arr = preds_arr * sc_covid + mn_covid
        acts_arr  = acts_arr  * sc_covid + mn_covid

    # Final Metrics
    if preds_arr.size > 0:
        preds_2d = preds_arr.reshape(-1, scaled_dataset.num_nodes)
        acts_2d  = acts_arr.reshape(-1, scaled_dataset.num_nodes)

        mae_per_node = mean_absolute_error(acts_2d, preds_2d, multioutput="raw_values")
        mse_per_node = mean_squared_error(acts_2d, preds_2d, multioutput="raw_values")
        rmse_per_node = np.sqrt(mse_per_node)
        r2_per_node   = r2_score(acts_2d, preds_2d, multioutput="raw_values")

        pcc_per_node = []
        for i in range(scaled_dataset.num_nodes):
            if np.std(acts_2d[:, i]) < 1e-6 or np.std(preds_2d[:, i]) < 1e-6:
                pcc_per_node.append(0.0)
            else:
                pcc_val, _ = pearsonr(acts_2d[:, i], preds_2d[:, i])
                if np.isnan(pcc_val):
                    pcc_val = 0.0
                pcc_per_node.append(pcc_val)
    else:
        mae_per_node = [float("nan")] * scaled_dataset.num_nodes
        rmse_per_node = [float("nan")] * scaled_dataset.num_nodes
        r2_per_node   = [float("nan")] * scaled_dataset.num_nodes
        pcc_per_node  = [float("nan")] * scaled_dataset.num_nodes

    # Store Metrics
    metrics_dict = {
        "Experiment_ID": [],
        "Adjacency_Type": [],
        "Horizon": [],
        "Region": [],
        "MAE": [],
        "RMSE": [],
        "R2_Score": [],
        "Pearson_Correlation": []
    }

    for i, region in enumerate(regions):
        metrics_dict["Experiment_ID"].append(experiment_id)
        metrics_dict["Adjacency_Type"].append(adjacency_type)
        metrics_dict["Horizon"].append(horizon)
        metrics_dict["Region"].append(region)
        metrics_dict["MAE"].append(mae_per_node[i])
        metrics_dict["RMSE"].append(rmse_per_node[i])
        metrics_dict["R2_Score"].append(r2_per_node[i])
        metrics_dict["Pearson_Correlation"].append(pcc_per_node[i])

        summary_metrics.append(
            {
                "Experiment_ID": experiment_id,
                "Adjacency_Type": adjacency_type,
                "Horizon": horizon,
                "Region": region,
                "MAE": mae_per_node[i],
                "RMSE": rmse_per_node[i],
                "R2_Score": r2_per_node[i],
                "Pearson_Correlation": pcc_per_node[i],
            }
        )

    # Save Metrics
    out_dir = os.path.join(project_root, "report", "metrics", f"experiment{experiment_id}_{adjacency_type}_h{horizon}")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"metrics_experiment{experiment_id}_{adjacency_type}_h{horizon}.csv")
    pd.DataFrame(metrics_dict).to_csv(out_path, index=False)
    logging.info(f"Metrics saved -> {out_path}")

    # [Optional] Additional Figure Generation Steps (Error Distributions, Actual vs Predicted, etc.)
    # You can implement similar plotting functions with shape checks as needed.

    # Save Final Model
    final_model_dir = os.path.join(project_root, "models", f"experiment{experiment_id}_{adjacency_type}_h{horizon}")
    os.makedirs(final_model_dir, exist_ok=True)
    final_model_path = os.path.join(final_model_dir, "epignn_final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    logging.info(f"Final model saved -> {final_model_path}")

    logging.info(f"--- Finished experiment {experiment_id}, adjacency={adjacency_type}, horizon={horizon} ---\n")

# ------------------------------------------------------------------------------
# Main: Load Data (Daily), Then Run Experiments
# ------------------------------------------------------------------------------
def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    setup_logging(project_root)
    seed_everything()

    # Create necessary directories
    os.makedirs(os.path.join(project_root, "figures"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "models"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "report", "metrics"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "report", "paper"), exist_ok=True)
    logging.info("Directories set up successfully.")

    # 1) Load daily data
    raw_path = os.path.join(project_root, "data", "raw", "merged_nhs_covid_data.csv")
    if not os.path.exists(raw_path):
        logging.error(f"CSV not found: {raw_path}")
        raise FileNotFoundError(raw_path)

    daily_data = pd.read_csv(raw_path, parse_dates=["date"])
    daily_data = load_and_correct_data_daily(daily_data, REFERENCE_COORDINATES)

    # 2) Save processed data
    proc_path = os.path.join(project_root, "data", "processed", "daily_nhs_covid_data.csv")
    os.makedirs(os.path.dirname(proc_path), exist_ok=True)
    daily_data.to_csv(proc_path, index=False)
    logging.info(f"Saved daily processed data -> {proc_path}")

    # 3) Define Horizons & Adjacency Types
    horizons = [3, 7, 14]
    adjacency_types = ["static", "dynamic", "hybrid"]
    experiment_id = 1
    summary_metrics = []

    # 4) Iterate Over Horizons and Adjacency Types
    for horizon in horizons:
        dataset = DailyDataset(
            data=daily_data,
            num_timesteps_input=NUM_TIMESTEPS_INPUT,
            num_timesteps_output=horizon,
            scaler=StandardScaler()
        )
        ds_len = len(dataset)
        logging.info(f"For horizon={horizon}, dataset length={ds_len}")

        if ds_len <= 0:
            logging.warning(f"No samples for horizon={horizon}. Skipping.")
            continue

        # 5) Train/Val/Test Splits
        total_len = ds_len
        train_size = int(0.7 * total_len)
        val_size   = int(0.15 * total_len)
        test_size  = total_len - train_size - val_size

        train_idx = list(range(0, train_size))
        val_idx   = list(range(train_size, train_size + val_size))
        test_idx  = list(range(train_size + val_size, total_len))

        train_subset = Subset(dataset, train_idx)
        val_subset   = Subset(dataset, val_idx)
        test_subset  = Subset(dataset, test_idx)

        logging.info(f"Horizon={horizon}, #Train={len(train_subset)}, #Val={len(val_subset)}, #Test={len(test_subset)}")

        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader   = DataLoader(val_subset,   batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        test_loader  = DataLoader(test_subset,  batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

        # 6) Compute Static Adjacency
        regions = dataset.regions.tolist()  # Changed from 'regs' to 'regions'
        latitudes  = [daily_data[daily_data["areaName"] == r]["latitude"].iloc[0] for r in regions]
        longitudes = [daily_data[daily_data["areaName"] == r]["longitude"].iloc[0] for r in regions]
        adj_static = compute_geographic_adjacency(regions, latitudes, longitudes, THRESHOLD_DISTANCE).to(device)

        logging.info("Static Adjacency Matrix:")
        logging.info(adj_static.cpu().numpy())

        # 7) Save Adjacency Figure
        fig_dir = os.path.join(project_root, "figures", f"experiment{experiment_id}_h{horizon}")
        os.makedirs(fig_dir, exist_ok=True)

        A_np = adj_static.cpu().numpy()
        G = nx.from_numpy_array(A_np)
        mapping = {i: r for i, r in enumerate(regions)}
        G = nx.relabel_nodes(G, mapping)
        pos = {r: (longitudes[i], latitudes[i]) for i, r in enumerate(regions)}  # Corrected variable name

        plt.figure(figsize=(10, 8))
        nx.draw_networkx(
            G,
            pos,
            with_labels=True,
            node_size=700,
            node_color="lightblue",
            edge_color="gray",
            font_size=10
        )
        plt.title("Static Adjacency (Geographic)", fontsize=12)
        plt.axis("off")
        out_figpath = os.path.join(fig_dir, "geographic_adjacency_graph_static.png")
        plt.savefig(out_figpath, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved adjacency figure -> {out_figpath}")

        # 8) Run Experiments for Each Adjacency Type
        for adj_type in adjacency_types:
            run_experiment(
                horizon=horizon,
                adjacency_type=adj_type,
                experiment_id=experiment_id,
                summary_metrics=summary_metrics,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                adj_static=adj_static,
                scaled_dataset=dataset,
                regions=regions,
                data=daily_data,
                test_subset=test_subset,
                train_size=train_size,
                val_size=val_size,
                project_root=project_root
            )
            experiment_id += 1

    # 9) Summarize All Experiments
    if len(summary_metrics) > 0:
        summary_df = pd.DataFrame(summary_metrics)
        summary_path = os.path.join(project_root, "report", "metrics", "summary_metrics.csv")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        summary_df.to_csv(summary_path, index=False)
        logging.info(f"Saved summary metrics -> {summary_path}")

        summary_pivot = summary_df.pivot_table(
            index=["Experiment_ID", "Adjacency_Type", "Horizon"],
            columns="Region",
            values=["MAE", "RMSE", "R2_Score", "Pearson_Correlation"]
        ).reset_index()

        pivot_path = os.path.join(project_root, "report", "metrics", "summary_metrics_pivot.csv")
        summary_pivot.to_csv(pivot_path, index=False)
        logging.info(f"Pivoted summary -> {pivot_path}")

        logging.info("All experiments completed. Summary of experiments:")
        logging.info(summary_pivot)
    else:
        logging.warning("No metrics collected. Possibly no data or skip conditions were triggered.")

if __name__ == "__main__":
    main()
