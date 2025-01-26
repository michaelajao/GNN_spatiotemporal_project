# train.py

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from haversine import haversine

# Local imports
from src.utils.utils import gravity_law_commute_dist
from model import STAN

# 1) Reproducibility
RANDOM_SEED = 123
def seed_torch(seed=RANDOM_SEED):
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

# 2) Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 3) Load or create data
raw_data = pd.read_csv('data/state_covid_data.csv')

pop_data = pd.read_csv('data/uszips.csv')
pop_data = pop_data.groupby('state_name').agg({
    'population':'sum',
    'density':'mean',
    'lat':'mean',
    'lng':'mean'
}).reset_index()

raw_data = pd.merge(raw_data, pop_data, how='inner', left_on='state', right_on='state_name')

loc_list = list(raw_data['state'].unique())
num_locations = len(loc_list)
print(f"Number of unique locations: {num_locations}")

# Build gravity-based adjacency
loc_dist_map = {}
for loc1 in loc_list:
    loc_dist_map[loc1] = {}
    lat1 = raw_data[raw_data['state'] == loc1]['lat'].unique()[0]
    lng1 = raw_data[raw_data['state'] == loc1]['lng'].unique()[0]
    pop1 = raw_data[raw_data['state'] == loc1]['population'].unique()[0]

    for loc2 in loc_list:
        lat2 = raw_data[raw_data['state'] == loc2]['lat'].unique()[0]
        lng2 = raw_data[raw_data['state'] == loc2]['lng'].unique()[0]
        pop2 = raw_data[raw_data['state'] == loc2]['population'].unique()[0]

        w = gravity_law_commute_dist(lat1, lng1, pop1, lat2, lng2, pop2, r=0.5)
        loc_dist_map[loc1][loc2] = w

# Sort connections and pick top few
for each_loc in loc_dist_map:
    loc_dist_map[each_loc] = {
        k: v for k, v in sorted(loc_dist_map[each_loc].items(), 
                               key=lambda item: item[1], 
                               reverse=True)
    }

dist_threshold = 18
adj_map = {}
for each_loc in loc_dist_map:
    adj_map[each_loc] = []
    for i, (each_loc2, distance) in enumerate(loc_dist_map[each_loc].items()):
        if distance > dist_threshold:
            # keep top 3
            if i < 3:
                adj_map[each_loc].append(each_loc2)
            else:
                break
        else:
            # keep top 1 if below threshold
            if i < 1:
                adj_map[each_loc].append(each_loc2)
            else:
                break

# Create global edge_index
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

# Add self-loops if desired:
self_loops = torch.arange(0, num_locations, dtype=torch.long).unsqueeze(0).repeat(2,1)
edge_index = torch.cat([edge_index, self_loops], dim=1)

# Move edge_index to device
edge_index = edge_index.to(device)

# Visualize the adjacency graph
G_nx = nx.DiGraph()
# Add nodes
for state in adj_map.keys():
    G_nx.add_node(state)

# Add edges
for state, neighbors in adj_map.items():
    for neighbor in neighbors:
        G_nx.add_edge(state, neighbor)

# Plot the adjacency graph
plt.figure(figsize=(15, 12))
pos = nx.spring_layout(G_nx, seed=RANDOM_SEED)  # For consistent layout
nx.draw(
    G_nx,
    pos,
    with_labels=True,
    node_size=3000,
    node_color='skyblue',
    font_size=10,
    font_weight='bold',
    edge_color='gray'
)
plt.title('Adjacency Map of States Based on Gravity Law')
plt.show()

# Visualize the adjacency heatmap
loc_dist_df = pd.DataFrame(loc_dist_map).fillna(0)

plt.figure(figsize=(15, 12))
sns.heatmap(loc_dist_df, cmap='viridis', linewidths=.5)
plt.title('Location Similarity Based on Gravity Law')
plt.xlabel('Location')
plt.ylabel('Location')
plt.show()

# Prepare standard S,I,R features
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
    popu   = df_loc['population'].values[0]
    dens   = df_loc['density'].values[0]
    lat    = df_loc['lat'].values[0]
    lng    = df_loc['lng'].values[0]

    recov = conf - active - death
    susc  = popu - active - recov  # naive S

    active_cases.append(active)
    recovered_cases.append(recov)
    death_cases.append(death)
    new_cases.append(newC)
    static_feat.append([popu, dens, lat, lng])
    susceptible_cases.append(susc)

active_cases      = np.array(active_cases)
recovered_cases   = np.array(recovered_cases)
susceptible_cases = np.array(susceptible_cases)
static_feat       = np.array(static_feat)

# Differences
dI = np.diff(active_cases, axis=1, prepend=0)
dR = np.diff(recovered_cases, axis=1, prepend=0)
dS = np.diff(susceptible_cases, axis=1, prepend=0)

# For your STAN, you used 3 dynamic features: [dI, dR, dS].
dynamic_feat = np.stack([dI, dR, dS], axis=-1)  # shape [num_loc, T, 3]

# Normalization (optional)
normalize = True
normalizer = {}
loc_mean_std = {}
for i, loc in enumerate(loc_list):
    mean_dI = dynamic_feat[i,:,0].mean()
    std_dI  = dynamic_feat[i,:,0].std() + 1e-5
    mean_dR = dynamic_feat[i,:,1].mean()
    std_dR  = dynamic_feat[i,:,1].std() + 1e-5
    mean_dS = dynamic_feat[i,:,2].mean()
    std_dS  = dynamic_feat[i,:,2].std() + 1e-5
    loc_mean_std[loc] = (mean_dI, std_dI, mean_dR, std_dR, mean_dS, std_dS)

if normalize:
    for i, loc in enumerate(loc_list):
        mI, sI, mR, sR, mS, sS = loc_mean_std[loc]
        dynamic_feat[i,:,0] = (dynamic_feat[i,:,0]-mI)/sI
        dynamic_feat[i,:,1] = (dynamic_feat[i,:,1]-mR)/sR
        dynamic_feat[i,:,2] = (dynamic_feat[i,:,2]-mS)/sS

# Train/val/test splits
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

def prepare_data(data, sum_I, sum_R, history_window=5, pred_window=15, slide_step=5):
    """
    Creates samples of:
      x:  shape [num_samples, num_loc, history_window * n_feat]
      I:  last known I
      R:  last known R
      yI: ground truth future I
      yR: ground truth future R
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

        # Last known
        lastI = sum_I[:, start_t+history_window-1]
        lastR = sum_R[:, start_t+history_window-1]
        lastI_all.append(lastI)
        lastR_all.append(lastR)

        # Future
        futureI = sum_I[:, start_t+history_window : start_t+history_window+pred_window]
        futureR = sum_R[:, start_t+history_window : start_t+history_window+pred_window]
        yI_all.append(futureI)
        yR_all.append(futureR)

    x_all     = np.array(x_all,     dtype=np.float32)  # [num_samples, n_loc, history_window*n_feat]
    lastI_all = np.array(lastI_all, dtype=np.float32)  # [num_samples, n_loc]
    lastR_all = np.array(lastR_all, dtype=np.float32)  # [num_samples, n_loc]
    yI_all    = np.array(yI_all,    dtype=np.float32)  # [num_samples, n_loc, pred_window]
    yR_all    = np.array(yR_all,    dtype=np.float32)

    return x_all, lastI_all, lastR_all, yI_all, yR_all

train_x, train_lastI, train_lastR, train_yI, train_yR = prepare_data(
    train_feat, train_I, train_R, 
    history_window, pred_window, slide_step=5
)
val_x, val_lastI, val_lastR, val_yI, val_yR = prepare_data(
    val_feat, val_I, val_R, 
    history_window, pred_window, slide_step=5
)
test_x, test_lastI, test_lastR, test_yI, test_yR = prepare_data(
    test_feat, test_I, test_R, 
    history_window, pred_window, slide_step=5
)

# Create a PyTorch Dataset
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
        # shapes:
        # x[idx] => [n_loc, history_window * n_feat]
        # lastI[idx], lastR[idx] => [n_loc]
        # yI[idx], yR[idx]       => [n_loc, pred_window]
        sample_x  = torch.tensor(self.x[idx], dtype=torch.float32)
        sample_I  = torch.tensor(self.lastI[idx], dtype=torch.float32)
        sample_R  = torch.tensor(self.lastR[idx], dtype=torch.float32)
        sample_yI = torch.tensor(self.yI[idx], dtype=torch.float32)
        sample_yR = torch.tensor(self.yR[idx], dtype=torch.float32)
        return sample_x, sample_I, sample_R, sample_yI, sample_yR

train_dataset = SIRDataset(train_x, train_lastI, train_lastR, train_yI, train_yR)
val_dataset   = SIRDataset(val_x,   val_lastI,   val_lastR,   val_yI,   val_yR)
test_dataset  = SIRDataset(test_x,  test_lastI,  test_lastR,  test_yI,  test_yR)

# Because your adjacency references 0..51, we do batch_size=1
batch_size = 1
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# Initialize STAN model
in_dim = 3 * history_window
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

# Create N as shape [num_locations,1] with the population
N = torch.tensor(static_feat[:,0], dtype=torch.float32, device=device).unsqueeze(-1)  # shape [52,1]

epoch_count = 50 if normalize else 300
scale = 0.1
best_val = float('inf')
file_name = './save/stan.pth'
os.makedirs('./save', exist_ok=True)

all_loss = []
for epoch in tqdm(range(epoch_count), desc='Training Epochs'):
    model.train()
    epoch_loss = 0.0

    for (batch_x, batch_I, batch_R, batch_yI, batch_yR) in train_loader:
        optimizer.zero_grad()

        # Move to device
        batch_x = batch_x.to(device)   # [1, 52, 18]
        batch_I = batch_I.to(device)   # [1,52]
        batch_R = batch_R.to(device)   # [1,52]
        batch_yI = batch_yI.to(device) # [1, 52, 15]
        batch_yR = batch_yR.to(device) # [1, 52, 15]

        # Reshape dynamic to [B, T, nLoc, F]
        B = batch_x.size(0)  # 1
        nLoc = batch_x.size(1)  # 52
        dynamic = batch_x.view(B, nLoc, history_window, 3)  # [1, 52, 6, 3]
        dynamic = dynamic.permute(0, 2, 1, 3).contiguous()  # [1,6,52,3]

        # states => shape [B*nLoc, 2]
        states = torch.cat([batch_I, batch_R], dim=-1).view(-1,2)  # [52, 2]

        predictions, phy_predictions = model(
            X=dynamic, 
            adj=edge_index, 
            states=states, 
            N=N
        )
        # predictions => [52, 15, 2]
        # batch_yI, batch_yR => [1, 52, 15]
        # Adjust predictions to match targets
        pred_I = predictions[:,:,0]  # [52, 15]
        pred_R = predictions[:,:,1]  # [52, 15]
        phy_I  = phy_predictions[:,:,0]  # [52, 15]
        phy_R  = phy_predictions[:,:,1]  # [52, 15]

        # If you have denormalization for phy_predictions, do it here
        # Example:
        # phy_I = phy_I * dI_std + dI_mean
        # phy_R = phy_R * dR_std + dR_mean

        # Loss
        loss = (criterion(pred_I, batch_yI.view(nLoc, pred_window)) +
                criterion(pred_R, batch_yR.view(nLoc, pred_window)) +
                scale * criterion(phy_I, batch_yI.view(nLoc, pred_window)) +
                scale * criterion(phy_R, batch_yR.view(nLoc, pred_window)))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss / len(train_loader)
    all_loss.append(epoch_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (batch_x, batch_I, batch_R, batch_yI, batch_yR) in val_loader:
            # Move to device
            batch_x = batch_x.to(device)
            batch_I = batch_I.to(device)
            batch_R = batch_R.to(device)
            batch_yI = batch_yI.to(device)
            batch_yR = batch_yR.to(device)

            B = batch_x.size(0)  # 1
            nLoc = batch_x.size(1)  # 52
            dynamic = batch_x.view(B, nLoc, history_window, 3).permute(0, 2, 1, 3).contiguous()  # [1,6,52,3]

            states = torch.cat([batch_I, batch_R], dim=-1).view(-1,2)  # [52,2]

            predictions, phy_predictions = model(
                X=dynamic, 
                adj=edge_index, 
                states=states, 
                N=N
            )

            pred_I = predictions[:,:,0]  # [52,15]
            pred_R = predictions[:,:,1]  # [52,15]
            phy_I  = phy_predictions[:,:,0]  # [52,15]
            phy_R  = phy_predictions[:,:,1]  # [52,15]

            loss = (criterion(pred_I, batch_yI.view(nLoc, pred_window)) +
                    criterion(pred_R, batch_yR.view(nLoc, pred_window)) +
                    scale * criterion(phy_I, batch_yI.view(nLoc, pred_window)) +
                    scale * criterion(phy_R, batch_yR.view(nLoc, pred_window)))
            val_loss += loss.item()

    val_loss = val_loss / len(val_loader)
    all_loss.append(epoch_loss)

    # Save the model if validation loss has decreased
    if val_loss < best_val:
        best_val = val_loss
        state = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, file_name)

    print(f"Epoch [{epoch+1}/{epoch_count}]  Train Loss: {epoch_loss:.4f}  Val Loss: {val_loss:.4f}")

# Plot training loss
plt.figure(figsize=(10,6))
plt.plot(all_loss, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

# Load best model
checkpoint = torch.load(file_name)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Test
test_preds_I = []
test_preds_R = []
test_phy_I = []
test_phy_R = []
test_truth_I = []
test_truth_R = []

with torch.no_grad():
    for (batch_x, batch_I, batch_R, batch_yI, batch_yR) in test_loader:
        # Move to device
        batch_x = batch_x.to(device)
        batch_I = batch_I.to(device)
        batch_R = batch_R.to(device)
        batch_yI = batch_yI.to(device)
        batch_yR = batch_yR.to(device)

        B = batch_x.size(0)  # 1
        nLoc = batch_x.size(1)  # 52
        dynamic = batch_x.view(B, nLoc, history_window, 3).permute(0, 2, 1, 3).contiguous()  # [1,6,52,3]

        states = torch.cat([batch_I, batch_R], dim=-1).view(-1,2)  # [52,2]

        predictions, phy_predictions = model(
            X=dynamic, 
            adj=edge_index, 
            states=states, 
            N=N
        )

        pred_I = predictions[:,:,0]  # [52,15]
        pred_R = predictions[:,:,1]  # [52,15]
        phy_I  = phy_predictions[:,:,0]  # [52,15]
        phy_R  = phy_predictions[:,:,1]  # [52,15]

        # Collect predictions
        test_preds_I.append(pred_I.cpu().numpy())
        test_preds_R.append(pred_R.cpu().numpy())
        test_phy_I.append(phy_I.cpu().numpy())
        test_phy_R.append(phy_R.cpu().numpy())
        test_truth_I.append(batch_yI.cpu().numpy())
        test_truth_R.append(batch_yR.cpu().numpy())

# Concatenate all batches
test_preds_I = np.concatenate(test_preds_I, axis=0)  # [num_samples * 52, 15]
test_preds_R = np.concatenate(test_preds_R, axis=0)
test_phy_I = np.concatenate(test_phy_I, axis=0)
test_phy_R = np.concatenate(test_phy_R, axis=0)
test_truth_I = np.concatenate(test_truth_I, axis=0)
test_truth_R = np.concatenate(test_truth_R, axis=0)

print("Test shape predictions I:", test_preds_I.shape)
print("Test shape predictions R:", test_preds_R.shape)
print("Test shape ground truth I:", test_truth_I.shape)
print("Test shape ground truth R:", test_truth_R.shape)

# Example: plot the first sample's predicted vs. ground truth curve for Active Cases
for loc_idx, loc_name in enumerate(loc_list):
    plt.figure(figsize=(12, 6))
    plt.plot(test_truth_I[loc_idx], label='Ground Truth I')
    plt.plot(test_preds_I[loc_idx], label='Predicted I')
    plt.plot(test_phy_I[loc_idx], label='Physical Model I', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Active Cases')
    plt.title(f'Active Cases: True vs Predicted for {loc_name}')
    plt.legend()
    plt.show()

# Similarly, plot for Recovered Cases
for loc_idx, loc_name in enumerate(loc_list):
    plt.figure(figsize=(12, 6))
    plt.plot(test_truth_R[loc_idx], label='Ground Truth R')
    plt.plot(test_preds_R[loc_idx], label='Predicted R')
    plt.plot(test_phy_R[loc_idx], label='Physical Model R', linestyle='--')
    plt.xlabel('Time Step')
    plt.ylabel('Recovered Cases')
    plt.title(f'Recovered Cases: True vs Predicted for {loc_name}')
    plt.legend()
    plt.show()

# Optionally, save predictions to a file
# np.save('./save/pred_I.npy', test_preds_I)
# np.save('./save/pred_R.npy', test_preds_R)
# np.save('./save/phy_I.npy', test_phy_I)
# np.save('./save/phy_R.npy', test_phy_R)
# np.save('./save/I_true.npy', test_truth_I)
# np.save('./save/R_true.npy', test_truth_R)
