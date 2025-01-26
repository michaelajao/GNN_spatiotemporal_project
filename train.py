# train.py

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from haversine import haversine

# Local imports
from src.utils.utils import gravity_law_commute_dist  # Ensure path is correct
from model import STAN  # Ensure model.py is in the same directory or adjust import path

###############################################################################
#                             1) Reproducibility                              #
###############################################################################

RANDOM_SEED = 123

def seed_torch(seed=RANDOM_SEED):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch()

###############################################################################
#                                 2) Device Setup                            #
###############################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

###############################################################################
#                           3) Load and Prepare Data                        #
###############################################################################

# 3.1 Load COVID-19 data
raw_data = pd.read_csv('data/state_covid_data.csv')

# 3.2 Load population and geographic data
pop_data = pd.read_csv('data/uszips.csv')
pop_data = pop_data.groupby('state_name').agg({
    'population': 'sum',
    'density': 'mean',
    'lat': 'mean',
    'lng': 'mean'
}).reset_index()

# 3.3 Merge with raw_data
raw_data = pd.merge(raw_data, pop_data, how='inner', left_on='state', right_on='state_name')

loc_list = list(raw_data['state'].unique())
num_locations = len(loc_list)
print(f"Number of unique locations: {num_locations}")

###############################################################################
#                         4) Build Gravity-Based Adjacency                  #
###############################################################################

loc_dist_map = {}
for loc1 in loc_list:
    loc_dist_map[loc1] = {}
    lat1 = float(raw_data[raw_data['state'] == loc1]['lat'].unique()[0])
    lng1 = float(raw_data[raw_data['state'] == loc1]['lng'].unique()[0])
    pop1 = float(raw_data[raw_data['state'] == loc1]['population'].unique()[0])

    for loc2 in loc_list:
        lat2 = float(raw_data[raw_data['state'] == loc2]['lat'].unique()[0])
        lng2 = float(raw_data[raw_data['state'] == loc2]['lng'].unique()[0])
        pop2 = float(raw_data[raw_data['state'] == loc2]['population'].unique()[0])

        w = gravity_law_commute_dist(lat1, lng1, pop1, lat2, lng2, pop2, r=0.5)
        loc_dist_map[loc1][loc2] = w

# Sort by descending distance; pick top neighbors based on threshold
dist_threshold = 18
adj_map = {}
for each_loc in loc_list:
    adj_map[each_loc] = []
    sorted_nbrs = sorted(loc_dist_map[each_loc].items(), key=lambda x: x[1], reverse=True)
    # Select top 3 neighbors above threshold, else top 1 below threshold
    for (nbr_loc, distance) in sorted_nbrs:
        if len(adj_map[each_loc]) >= 3:
            break
        if distance > dist_threshold:
            # top 3 above threshold
            adj_map[each_loc].append(nbr_loc)
        else:
            # top 1 below threshold if no neighbor added yet
            if len(adj_map[each_loc]) < 1:
                adj_map[each_loc].append(nbr_loc)

    # If after all that, still no neighbors, add self-loop
    if len(adj_map[each_loc]) == 0:
        adj_map[each_loc].append(each_loc)

###############################################################################
#                         5) Create Edge Index Tensor                       #
###############################################################################

state_to_index = {s: idx for idx, s in enumerate(loc_list)}
edge_src = []
edge_dst = []
for st, nbrs in adj_map.items():
    for nb in nbrs:
        edge_src.append(state_to_index[st])
        edge_dst.append(state_to_index[nb])

edge_src = torch.tensor(edge_src, dtype=torch.long)
edge_dst = torch.tensor(edge_dst, dtype=torch.long)
edge_index = torch.stack([edge_src, edge_dst], dim=0)

# Add self-loops
self_loops = torch.arange(0, num_locations, dtype=torch.long).unsqueeze(0).repeat(2,1)
edge_index = torch.cat([edge_index, self_loops], dim=1)

edge_index = edge_index.to(device)

###############################################################################
#                         6) Visualization Functions                       #
###############################################################################

def visualize_adjacency(adj_map, loc_list, state_to_index, static_feat):
    """
    Draws adjacency graph of states using gravity law connections.
    """
    G_nx = nx.DiGraph()
    for state in adj_map.keys():
        idx = state_to_index[state]
        population = static_feat[idx, 0]
        density = static_feat[idx, 1]
        G_nx.add_node(state, population=population, density=density)

    for state, neighbors in adj_map.items():
        for neighbor in neighbors:
            G_nx.add_edge(state, neighbor)

    populations = [G_nx.nodes[node]['population'] for node in G_nx.nodes()]
    max_pop = max(populations) if len(populations) > 0 else 1
    node_sizes = [300 + (pop / max_pop) * 1700 for pop in populations]

    densities = [G_nx.nodes[node]['density'] for node in G_nx.nodes()]
    cmap = plt.cm.viridis
    norm = plt.Normalize(min(densities), max(densities))
    node_colors = cmap(norm(densities))

    plt.figure(figsize=(20, 16))
    pos = nx.spring_layout(G_nx, seed=RANDOM_SEED, k=0.15, iterations=20)
    nx.draw_networkx_nodes(G_nx, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G_nx, pos, arrowstyle='->', arrowsize=15, edge_color='gray', alpha=0.5)
    nx.draw_networkx_labels(G_nx, pos, font_size=10, font_weight='bold')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(densities)
    cbar = plt.colorbar(sm, ax=plt.gca(), shrink=0.5, aspect=20)

    plt.title('Adjacency Map of States Based on Gravity Law')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('adjacency_map_enhanced.png', dpi=300)
    plt.show()

def visualize_heatmap(loc_dist_map, loc_list):
    """
    Draws heatmap of location similarities using gravity law.
    """
    loc_dist_df = pd.DataFrame(loc_dist_map).reindex(loc_list).fillna(0)
    plt.figure(figsize=(20, 16))
    sns.heatmap(loc_dist_df, cmap='viridis', linewidths=.5, 
                xticklabels=loc_list, yticklabels=loc_list, annot=False)
    plt.title('Location Similarity Based on Gravity Law')
    plt.xlabel('Location')
    plt.ylabel('Location')
    plt.tight_layout()
    plt.savefig('adjacency_heatmap_enhanced.png', dpi=300)
    plt.show()

###############################################################################
#                          7) Build Static Features                          #
###############################################################################

print("Building static features...")
active_cases = []
recovered_cases = []
susceptible_cases = []
death_cases = []
new_cases = []
static_feat = []

for each_loc in loc_list:
    df_loc = raw_data[raw_data['state'] == each_loc]
    active = df_loc['active'].values
    conf   = df_loc['confirmed'].values
    death  = df_loc['deaths'].values
    newC   = df_loc['new_cases'].values
    popu   = float(df_loc['population'].values[0])
    dens   = float(df_loc['density'].values[0])
    lat    = float(df_loc['lat'].values[0])
    lng    = float(df_loc['lng'].values[0])

    recov = conf - active - death
    susc  = popu - active - recov  # naive susceptible

    active_cases.append(active)
    recovered_cases.append(recov)
    susceptible_cases.append(susc)  # critical to fill
    death_cases.append(death)
    new_cases.append(newC)
    static_feat.append([popu, dens, lat, lng])

active_cases      = np.array(active_cases)
recovered_cases   = np.array(recovered_cases)
susceptible_cases = np.array(susceptible_cases)
static_feat       = np.array(static_feat)

print(f"Static features shape: {static_feat.shape}") 
# e.g. [52,4]

# 7.1 Compute daily differences
dI = np.diff(active_cases, axis=1, prepend=0)
dR = np.diff(recovered_cases, axis=1, prepend=0)
dS = np.diff(susceptible_cases, axis=1, prepend=0)

print(f"dI shape: {dI.shape}")
print(f"dR shape: {dR.shape}")
print(f"dS shape: {dS.shape}")

dynamic_feat = np.stack([dI, dR, dS], axis=-1)
print(f"Dynamic features shape: {dynamic_feat.shape}")

# 7.2 Visualize adjacency and heatmap
visualize_adjacency(adj_map, loc_list, state_to_index, static_feat)
visualize_heatmap(loc_dist_map, loc_list)

###############################################################################
#                         8) Normalize Dynamic Features                      #
###############################################################################

normalize = True
loc_mean_std = {}

if normalize:
    print("\nNormalizing dynamic features...")
    for i, loc in enumerate(loc_list):
        mean_dI = dynamic_feat[i,:,0].mean()
        std_dI  = dynamic_feat[i,:,0].std() + 1e-5
        mean_dR = dynamic_feat[i,:,1].mean()
        std_dR  = dynamic_feat[i,:,1].std() + 1e-5
        mean_dS = dynamic_feat[i,:,2].mean()
        std_dS  = dynamic_feat[i,:,2].std() + 1e-5
        loc_mean_std[loc] = (mean_dI, std_dI, mean_dR, std_dR, mean_dS, std_dS)

    for i, loc in enumerate(loc_list):
        mI, sI, mR, sR, mS, sS = loc_mean_std[loc]
        dynamic_feat[i,:,0] = (dynamic_feat[i,:,0] - mI) / sI
        dynamic_feat[i,:,1] = (dynamic_feat[i,:,1] - mR) / sR
        dynamic_feat[i,:,2] = (dynamic_feat[i,:,2] - mS) / sS

###############################################################################
#                         9) Train/Val/Test Splits                          #
###############################################################################

valid_window = 25
test_window  = 25
history_window = 6
pred_window = 15
slide_step  = 5

train_feat = dynamic_feat[:, :-valid_window - test_window, :]
val_feat   = dynamic_feat[:, -valid_window - test_window:-test_window, :]
test_feat  = dynamic_feat[:, -test_window:, :]

train_I = active_cases[:, :-valid_window - test_window]
train_R = recovered_cases[:, :-valid_window - test_window]

val_I = active_cases[:, -valid_window - test_window:-test_window]
val_R = recovered_cases[:, -valid_window - test_window:-test_window]

test_I = active_cases[:, -test_window:]
test_R = recovered_cases[:, -test_window:]

###############################################################################
#                         10) Data Preparation Function                     #
###############################################################################

def prepare_data(data, sum_I, sum_R, history_window=6, pred_window=15, slide_step=5):
    """
    Creates samples for the model using a sliding window approach.

    :param data: [num_loc, T, 3] 
    :param sum_I: [num_loc, T]
    :param sum_R: [num_loc, T]
    :return: x, lastI, lastR, yI, yR
    """
    n_loc = data.shape[0]
    T     = data.shape[1]
    n_feat= data.shape[2]
    x_all = []
    lastI_all = []
    lastR_all = []
    yI_all = []
    yR_all = []

    for start_t in range(0, T - history_window - pred_window +1, slide_step):
        x_slice = data[:, start_t:start_t+history_window, :].reshape(n_loc, history_window*n_feat)
        x_all.append(x_slice)

        # Last known I, R
        lastI = sum_I[:, start_t+history_window-1]
        lastR = sum_R[:, start_t+history_window-1]
        lastI_all.append(lastI)
        lastR_all.append(lastR)

        # Future I, R
        futureI = sum_I[:, start_t+history_window : start_t+history_window+pred_window]
        futureR = sum_R[:, start_t+history_window : start_t+history_window+pred_window]
        yI_all.append(futureI)
        yR_all.append(futureR)

    x_all     = np.array(x_all,     dtype=np.float32)
    lastI_all = np.array(lastI_all, dtype=np.float32)
    lastR_all = np.array(lastR_all, dtype=np.float32)
    yI_all    = np.array(yI_all,    dtype=np.float32)
    yR_all    = np.array(yR_all,    dtype=np.float32)

    return x_all, lastI_all, lastR_all, yI_all, yR_all

print("Preparing train, validation, and test sets...")
train_x, train_lastI, train_lastR, train_yI, train_yR = prepare_data(
    train_feat, train_I, train_R, history_window, pred_window, slide_step
)
val_x, val_lastI, val_lastR, val_yI, val_yR = prepare_data(
    val_feat, val_I, val_R, history_window, pred_window, slide_step
)
test_x, test_lastI, test_lastR, test_yI, test_yR = prepare_data(
    test_feat, test_I, test_R, history_window, pred_window, slide_step
)

print(f"Train set shape: x={train_x.shape}, yI={train_yI.shape}, yR={train_yR.shape}")
print(f"Validation set shape: x={val_x.shape}, yI={val_yI.shape}, yR={val_yR.shape}")
print(f"Test set shape: x={test_x.shape}, yI={test_yI.shape}, yR={test_yR.shape}")

###############################################################################
#                         11) Create PyTorch Dataset                        #
###############################################################################

class SIRDataset(Dataset):
    def __init__(self, x, lastI, lastR, yI, yR):
        self.x     = x
        self.lastI = lastI
        self.lastR = lastR
        self.yI    = yI
        self.yR    = yR

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (torch.tensor(self.x[idx], dtype=torch.float32),
                torch.tensor(self.lastI[idx], dtype=torch.float32),
                torch.tensor(self.lastR[idx], dtype=torch.float32),
                torch.tensor(self.yI[idx], dtype=torch.float32),
                torch.tensor(self.yR[idx], dtype=torch.float32))

train_dataset = SIRDataset(train_x, train_lastI, train_lastR, train_yI, train_yR)
val_dataset   = SIRDataset(val_x,   val_lastI,   val_lastR,   val_yI,   val_yR)
test_dataset  = SIRDataset(test_x,  test_lastI,  test_lastR,  test_yI,  test_yR)

###############################################################################
#                          12) Create DataLoaders                             #
###############################################################################

batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

###############################################################################
#                             13) Initialize Model                            #
###############################################################################

in_dim = 3 * history_window  # 3 features * 6 history window
model = STAN(
    num_nodes=num_locations,
    num_features=in_dim,
    num_timesteps_input=history_window,
    num_timesteps_output=pred_window,
    population=1e10,
    gat_dim1=32,
    gat_dim2=32,
    gru_dim=32,
    num_heads=1,
    device=device
).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.MSELoss()

N = torch.tensor(static_feat[:,0], dtype=torch.float32, device=device).unsqueeze(-1)

###############################################################################
#                        14) Training Parameters                             #
###############################################################################

epoch_count = 50 if normalize else 300
scale = 0.1
best_val = float('inf')
file_name = './save/stan.pth'
os.makedirs('./save', exist_ok=True)

train_losses = []
val_losses   = []

###############################################################################
#                                15) Training Loop                             #
###############################################################################

for epoch in tqdm(range(epoch_count), desc='Training Epochs'):
    model.train()
    epoch_loss = 0.0

    for (batch_x, batch_I, batch_R, batch_yI, batch_yR) in train_loader:
        optimizer.zero_grad()

        batch_x, batch_I, batch_R, batch_yI, batch_yR = (
            batch_x.to(device),
            batch_I.to(device),
            batch_R.to(device),
            batch_yI.to(device),
            batch_yR.to(device)
        )

        B    = batch_x.size(0) 
        nLoc = batch_x.size(1)
        dynamic = batch_x.view(B, nLoc, history_window, 3).permute(0, 2, 1, 3).contiguous()
        states = torch.cat([batch_I, batch_R], dim=-1).view(-1, 2)

        # Forward
        predictions, phy_predictions = model(
            X=dynamic, 
            adj=edge_index, 
            states=states, 
            N=N
        )
        # predictions => [52, 15, 2]

        pred_I = predictions[:,:,0]
        pred_R = predictions[:,:,1]
        phy_I  = phy_predictions[:,:,0]
        phy_R  = phy_predictions[:,:,1]

        # Normalize targets for MSE
        yI_normalized = []
        yR_normalized = []
        for i, loc_name in enumerate(loc_list):
            mI, sI, mR, sR, _, _ = loc_mean_std[loc_name]
            yI_norm = (batch_yI[0, i, :].cpu().numpy() - mI) / sI
            yR_norm = (batch_yR[0, i, :].cpu().numpy() - mR) / sR
            yI_normalized.append(yI_norm)
            yR_normalized.append(yR_norm)
        yI_normalized = torch.tensor(yI_normalized, dtype=torch.float32, device=device)
        yR_normalized = torch.tensor(yR_normalized, dtype=torch.float32, device=device)

        loss = (criterion(pred_I, yI_normalized) +
                criterion(pred_R, yR_normalized) +
                scale * criterion(phy_I, yI_normalized) +
                scale * criterion(phy_R, yR_normalized))

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader)
    train_losses.append(epoch_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (batch_x, batch_I, batch_R, batch_yI, batch_yR) in val_loader:
            batch_x, batch_I, batch_R, batch_yI, batch_yR = (
                batch_x.to(device),
                batch_I.to(device),
                batch_R.to(device),
                batch_yI.to(device),
                batch_yR.to(device)
            )

            B    = batch_x.size(0)
            nLoc = batch_x.size(1)
            dynamic = batch_x.view(B, nLoc, history_window, 3).permute(0, 2, 1, 3).contiguous()
            states = torch.cat([batch_I, batch_R], dim=-1).view(-1, 2)

            # Forward
            predictions, phy_predictions = model(
                X=dynamic, 
                adj=edge_index, 
                states=states, 
                N=N
            )

            pred_I = predictions[:,:,0]
            pred_R = predictions[:,:,1]
            phy_I  = phy_predictions[:,:,0]
            phy_R  = phy_predictions[:,:,1]

            # Normalize targets
            yI_normalized = []
            yR_normalized = []
            for i, loc_name in enumerate(loc_list):
                mI, sI, mR, sR, _, _ = loc_mean_std[loc_name]
                yI_norm = (batch_yI[0, i, :].cpu().numpy() - mI) / sI
                yR_norm = (batch_yR[0, i, :].cpu().numpy() - mR) / sR
                yI_normalized.append(yI_norm)
                yR_normalized.append(yR_norm)
            yI_normalized = torch.tensor(yI_normalized, dtype=torch.float32, device=device)
            yR_normalized = torch.tensor(yR_normalized, dtype=torch.float32, device=device)

            loss = (criterion(pred_I, yI_normalized) +
                    criterion(pred_R, yR_normalized) +
                    scale * criterion(phy_I, yI_normalized) +
                    scale * criterion(phy_R, yR_normalized))
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    # Save best
    if val_loss < best_val:
        best_val = val_loss
        torch.save({
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, file_name)

    print(f"Epoch [{epoch+1}/{epoch_count}]  Train Loss: {epoch_loss:.4f}  Val Loss: {val_loss:.4f}")

###############################################################################
#                           16) Plot Training Losses                          #
###############################################################################

plt.figure(figsize=(10,6))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('training_validation_loss.png', dpi=300)
plt.show()

###############################################################################
#                             17) Load Best Model                             #
###############################################################################

checkpoint = torch.load(file_name)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

###############################################################################
#                             18) Testing and Evaluation                      #
###############################################################################

test_preds_I = []
test_preds_R = []
test_phy_I   = []
test_phy_R   = []
test_truth_I = []
test_truth_R = []

with torch.no_grad():
    for (batch_x, batch_I, batch_R, batch_yI, batch_yR) in test_loader:
        batch_x, batch_I, batch_R, batch_yI, batch_yR = (
            batch_x.to(device),
            batch_I.to(device),
            batch_R.to(device),
            batch_yI.to(device),
            batch_yR.to(device)
        )

        B    = batch_x.size(0)
        nLoc = batch_x.size(1)
        dynamic = batch_x.view(B, nLoc, history_window, 3).permute(0, 2, 1, 3).contiguous()
        states = torch.cat([batch_I, batch_R], dim=-1).view(-1, 2)

        predictions, phy_predictions = model(X=dynamic, adj=edge_index, states=states, N=N)
        pred_I = predictions[:,:,0]
        pred_R = predictions[:,:,1]
        phy_I  = phy_predictions[:,:,0]
        phy_R  = phy_predictions[:,:,1]

        # Denormalize predictions
        pred_I_denorm = []
        pred_R_denorm = []
        phy_I_denorm  = []
        phy_R_denorm  = []
        yI_denorm     = []
        yR_denorm     = []
        for i, loc_name in enumerate(loc_list):
            mI, sI, mR, sR, _, _ = loc_mean_std[loc_name]
            pred_I_denorm.append(pred_I[i].cpu().numpy() * sI + mI)
            pred_R_denorm.append(pred_R[i].cpu().numpy() * sR + mR)
            phy_I_denorm.append(phy_I[i].cpu().numpy() * sI + mI)
            phy_R_denorm.append(phy_R[i].cpu().numpy() * sR + mR)
            yI_denorm.append(batch_yI[0, i, :].cpu().numpy())
            yR_denorm.append(batch_yR[0, i, :].cpu().numpy())

        test_preds_I.append(np.array(pred_I_denorm))
        test_preds_R.append(np.array(pred_R_denorm))
        test_phy_I.append(np.array(phy_I_denorm))
        test_phy_R.append(np.array(phy_R_denorm))
        test_truth_I.append(np.array(yI_denorm))
        test_truth_R.append(np.array(yR_denorm))

test_preds_I = np.concatenate(test_preds_I, axis=0)
test_preds_R = np.concatenate(test_preds_R, axis=0)
test_phy_I   = np.concatenate(test_phy_I, axis=0)
test_phy_R   = np.concatenate(test_phy_R, axis=0)
test_truth_I = np.concatenate(test_truth_I, axis=0)
test_truth_R = np.concatenate(test_truth_R, axis=0)

print("Test Predictions I shape:", test_preds_I.shape)
print("Test Predictions R shape:", test_preds_R.shape)
print("Test Ground Truth I shape:", test_truth_I.shape)
print("Test Ground Truth R shape:", test_truth_R.shape)

###############################################################################
#                          19) Visualization Function                        #
###############################################################################

def visualize_predictions(test_preds, test_phy, test_truth, loc_list, case_type='Active Cases', samples_to_plot=5):
    """
    Plots predictions vs ground truth for a subset of locations.
    """
    num_locations = len(loc_list)
    samples_to_plot = min(samples_to_plot, num_locations)

    plt.figure(figsize=(20, 4 * samples_to_plot))
    for i in range(samples_to_plot):
        plt.subplot(samples_to_plot, 1, i+1)
        plt.plot(test_truth[i], label='Ground Truth', color='black')
        plt.plot(test_preds[i], label='Predicted', color='blue')
        plt.plot(test_phy[i], label='Physical Model', color='red', linestyle='--')
        plt.xlabel('Time Step')
        plt.ylabel(case_type)
        plt.title(f'{case_type}: {loc_list[i]}')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{case_type.lower().replace(" ", "_")}_predictions.png', dpi=300)
    plt.show()

###############################################################################
#                        20) Visualize Final Predictions                     #
###############################################################################

visualize_predictions(test_preds_I, test_phy_I, test_truth_I, loc_list, 'Active Cases', samples_to_plot=5)
visualize_predictions(test_preds_R, test_phy_R, test_truth_R, loc_list, 'Recovered Cases', samples_to_plot=5)

###############################################################################
#                          21) (Optional) Save Predictions                   #
###############################################################################

# np.save('./save/pred_I.npy', test_preds_I)
# np.save('./save/pred_R.npy', test_preds_R)
# np.save('./save/phy_I.npy', test_phy_I)
# np.save('./save/phy_R.npy', test_phy_R)
# np.save('./save/I_true.npy', test_truth_I)
# np.save('./save/R_true.npy', test_truth_R)
