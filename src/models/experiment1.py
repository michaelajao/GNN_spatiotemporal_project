import os
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from haversine import haversine
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
import networkx as nx
import matplotlib.animation as animation
from epiweeks import Week

# =======================
# Step 1: Define Helper Functions
# =======================

def gravity_law_commute_dist(lat1, lng1, pop1, lat2, lng2, pop2, r=1e4):
    """
    Calculate the gravity law edge weight between two locations.

    Parameters:
    - lat1, lng1: Latitude and longitude of the first location.
    - pop1: Population of the first location.
    - lat2, lng2: Latitude and longitude of the second location.
    - pop2: Population of the second location.
    - r: Scaling factor (default: 10,000 km).

    Returns:
    - Edge weight calculated based on the gravity law.
    """
    d = haversine((lat1, lng1), (lat2, lng2))  # Distance in kilometers
    alpha = 0.1
    beta = 0.1
    w = (np.exp(-d / r)) / (abs((pop1 ** alpha) - (pop2 ** beta)) + 1e-5)
    return w

def gravity_law_commute_dist_dynamic(lat1, lng1, pop1, lat2, lng2, pop2, dynamic_feature1, dynamic_feature2, r=1e4):
    """
    Calculate the dynamic gravity law edge weight between two locations.

    Parameters:
    - lat1, lng1: Latitude and longitude of the first location.
    - pop1: Population of the first location.
    - lat2, lng2: Latitude and longitude of the second location.
    - pop2: Population of the second location.
    - dynamic_feature1: Dynamic feature (e.g., new_cases) of the first location.
    - dynamic_feature2: Dynamic feature of the second location.
    - r: Scaling factor (default: 10,000 km).

    Returns:
    - Edge weight calculated based on the dynamic gravity law.
    """
    d = haversine((lat1, lng1), (lat2, lng2))  # Distance in kilometers
    alpha = 0.1
    beta = 0.1
    dynamic_factor = (dynamic_feature1 + dynamic_feature2) / 2
    w = (np.exp(-d / r)) * dynamic_factor / (abs((pop1 ** alpha) - (pop2 ** beta)) + 1e-5)
    return w

def map_to_week(df, date_column='date_today', groupby_target=None):
    """
    Map dates to week ending dates and aggregate data accordingly.

    Parameters:
    - df: Pandas DataFrame containing the data.
    - date_column: Name of the column containing dates.
    - groupby_target: List of columns to aggregate.

    Returns:
    - DataFrame with dates mapped to weeks and aggregated.
    """
    df[date_column] = df[date_column].apply(
        lambda x: Week.fromdate(x).enddate() if pd.notna(x) else x
    )
    df[date_column] = pd.to_datetime(df[date_column])

    if groupby_target is not None:
        df = df.groupby([date_column, 'state'], as_index=False)[groupby_target].sum()
    return df

# =======================
# Step 2: Set Parameters
# =======================

K_NEIGHBORS = 5  # Number of nearest neighbors for each state
START_DATE = '2020-06-01'
<<<<<<< HEAD
END_DATE = '2022-12-01'
=======
END_DATE = '2021-12-01'
>>>>>>> 9044e3b99f5c8fbfc2afb13e9b6e1b5bc127319c
FEATURE_COLUMNS = ['confirmed', 'deaths', 'recovered', 'active', 'hospitalization', 'new_cases']

# =======================
# Step 3: Load and Preprocess Data
# =======================

# 3.1 Load COVID-19 Data
covid_data_path = '../../data/processed/processed_covid_data.pickle'
with open(covid_data_path, 'rb') as f:
    raw_data = pickle.load(f)

# 3.2 Load Population Data
pop_data_path = '../../data/uszips.csv'
pop_data = pd.read_csv(pop_data_path)

# 3.3 Aggregate Population Data by State
pop_data = pop_data.groupby('state_name').agg({
    'population': 'sum',
    'density': 'mean',
    'lat': 'mean',
    'lng': 'mean'
}).reset_index()

# 3.4 Merge COVID Data with Population Data
raw_data = pd.merge(
    raw_data,
    pop_data,
    how='inner',
    left_on='state',
    right_on='state_name'
)

# 3.5 Convert 'date_today' Column to Datetime
raw_data['date_today'] = pd.to_datetime(raw_data['date_today'])

# 3.6 Handle Missing Values in 'hospitalization' Column
raw_data['hospitalization'] = raw_data['hospitalization'].ffill()
raw_data['hospitalization'] = raw_data['hospitalization'].fillna(0)

# Verify that there are no NaNs left in 'hospitalization'
num_nans_hosp = raw_data['hospitalization'].isna().sum()
print(f"Number of NaNs in 'hospitalization' after filling: {num_nans_hosp}")

# 3.7 Filter Data Between START_DATE and END_DATE
raw_data = raw_data[
    (raw_data['date_today'] >= pd.to_datetime(START_DATE)) &
    (raw_data['date_today'] <= pd.to_datetime(END_DATE))
].reset_index(drop=True)

# 3.8 Map Dates to Weeks and Aggregate Data
raw_data = map_to_week(raw_data, date_column='date_today', groupby_target=FEATURE_COLUMNS)

# 3.9 Re-Merge Population Data (Optional)
state_population = pop_data[['state_name', 'population', 'density', 'lat', 'lng']].rename(columns={'state_name': 'state'})
raw_data = pd.merge(raw_data, state_population, on='state', how='left')

# Handle any NaNs that might have arisen from the merge
raw_data[FEATURE_COLUMNS] = raw_data[FEATURE_COLUMNS].fillna(0)

# =======================
# Step 4: Create State Information
# =======================

# 4.1 Extract Unique States
states = raw_data['state'].unique()

# 4.2 Create a DataFrame with State Information
state_info = state_population[state_population['state'].isin(states)].reset_index(drop=True)

# 4.3 Assign Unique Node IDs to States
state_info = state_info.reset_index().rename(columns={'index': 'node_id'})
state_info['node_id'] = state_info['node_id'].astype(int)

# 4.4 Create a Mapping from State Name to Node ID
state_to_id = dict(zip(state_info['state'], state_info['node_id']))

# 4.5 Verify Uniqueness of States
duplicate_states = state_info['state'].duplicated().sum()
if duplicate_states > 0:
    raise ValueError(f"There are {duplicate_states} duplicate states in 'state_info'. Ensure all states are unique.")
else:
    print("All states in 'state_info' are unique.")

# =======================
# Step 5: Static Graph Construction
# =======================

# 5.1 Compute Pairwise Distances Between States
num_states = len(state_info)
distance_matrix = np.zeros((num_states, num_states))

# Extract latitude and longitude as numpy arrays for efficient computation
latitudes = state_info['lat'].values
longitudes = state_info['lng'].values

# Compute pairwise distances using Haversine formula
print("Computing pairwise distances for static graph...")
for i in tqdm(range(num_states), desc="Static Distance Matrix"):
    for j in range(num_states):
        if i == j:
            distance_matrix[i, j] = 0
        else:
            distance = haversine((latitudes[i], longitudes[i]), (latitudes[j], longitudes[j]))
            distance_matrix[i, j] = distance

# 5.2 Define Edges Based on K Nearest Neighbors for Static Graph
print("Defining edges for static graph...")
neighbors = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric='precomputed')  # +1 to include self
neighbors.fit(distance_matrix)
distances, indices = neighbors.kneighbors(distance_matrix)

# 5.3 Initialize Lists to Hold Edge Indices and Weights
static_edge_index = []
static_edge_weight = []

# 5.4 Iterate Over Each State to Define Static Edges
for i in tqdm(range(num_states), desc="Static Edge Definition"):
    for j in range(1, K_NEIGHBORS + 1):  # Start from 1 to exclude self-loop
        neighbor = indices[i, j]
        static_edge_index.append([i, neighbor])

        # Retrieve populations and coordinates for the gravity law calculation
        pop_i = state_info.loc[i, 'population']
        pop_j = state_info.loc[neighbor, 'population']
        lat_i = state_info.loc[i, 'lat']
        lng_i = state_info.loc[i, 'lng']
        lat_j = state_info.loc[neighbor, 'lat']
        lng_j = state_info.loc[neighbor, 'lng']

        # Calculate edge weight using gravity law
        weight = gravity_law_commute_dist(lat_i, lng_i, pop_i, lat_j, lng_j, pop_j)
        static_edge_weight.append(weight)

# 5.5 Convert Edge Lists to Tensors
static_edge_index = torch.tensor(static_edge_index, dtype=torch.long).t().contiguous()
static_edge_weight = torch.tensor(static_edge_weight, dtype=torch.float)

# 5.6 Make the Static Graph Undirected by Adding Reverse Edges
static_edge_index = torch.cat([static_edge_index, static_edge_index[[1, 0], :]], dim=1)
static_edge_weight = torch.cat([static_edge_weight, static_edge_weight], dim=0)

# =======================
# Step 6: Prepare Temporal Graph Snapshots
# =======================

# 6.1 Get All Unique Weeks Sorted
all_weeks = raw_data['date_today'].sort_values().unique()
all_weeks = pd.to_datetime(all_weeks)

# 6.2 Initialize Lists to Hold Graph Snapshots
static_graph_snapshots = []
dynamic_graph_snapshots = []

# 6.3 Normalize Node Features Using GroupBy and Transform
raw_data_norm = raw_data.copy()

for feature in FEATURE_COLUMNS:
    # Calculate min and max per state
    feature_min = raw_data_norm.groupby('state')[feature].transform('min')
    feature_max = raw_data_norm.groupby('state')[feature].transform('max')

    # Apply min-max normalization
    raw_data_norm[feature] = (raw_data_norm[feature] - feature_min) / (feature_max - feature_min + 1e-8)

# 6.4 Verify Normalization Does Not Introduce NaNs
num_nans_norm = raw_data_norm[FEATURE_COLUMNS].isna().sum().sum()
print(f"Number of NaNs after normalization: {num_nans_norm}")

# 6.5 Iterate Over Each Week to Create Graph Snapshots
print("Creating static and dynamic graph snapshots...")
for current_week in tqdm(all_weeks, desc="Graph Snapshots"):
    weekly_data = raw_data_norm[raw_data_norm['date_today'] == current_week]

    # Ensure the order of states matches between weekly_data and state_info
    weekly_data = weekly_data.set_index('state').loc[state_info['state']].reset_index()

    # Check for any NaNs in node features
    if weekly_data[FEATURE_COLUMNS].isna().any().any():
        print(f"NaNs found in node features on {current_week}. Handling them.")
        # Handle NaNs by filling with zero
        weekly_data[FEATURE_COLUMNS] = weekly_data[FEATURE_COLUMNS].fillna(0)

    # Extract node features as a tensor
    node_features = torch.tensor(weekly_data[FEATURE_COLUMNS].values, dtype=torch.float)

    # -------------------------------
    # Static Graph Snapshot
    # -------------------------------
    data_static = Data(
        x=node_features,                # Node features
        edge_index=static_edge_index,    # Static edge indices
        edge_attr=static_edge_weight    # Static edge weights
    )
    data_static.date = current_week    # Add the week date as metadata
    static_graph_snapshots.append(data_static)

    # -------------------------------
    # Dynamic Graph Snapshot
    # -------------------------------
    # Dynamic Edge Construction
    edge_index_dynamic = []
    edge_weight_dynamic = []

    # Extract dynamic feature for edge weight calculation
    # Example: 'new_cases' as the dynamic feature
    dynamic_feature = weekly_data['new_cases'].values

    for i in range(num_states):
        for j in range(num_states):
            if i != j:
                lat_i = state_info.loc[i, 'lat']
                lng_i = state_info.loc[i, 'lng']
                pop_i = state_info.loc[i, 'population']
                lat_j = state_info.loc[j, 'lat']
                lng_j = state_info.loc[j, 'lng']
                pop_j = state_info.loc[j, 'population']

                # Dynamic features for states i and j
                dyn_feat_i = dynamic_feature[i]
                dyn_feat_j = dynamic_feature[j]

                # Calculate edge weight using dynamic gravity law
                weight = gravity_law_commute_dist_dynamic(
                    lat_i, lng_i, pop_i,
                    lat_j, lng_j, pop_j,
                    dyn_feat_i, dyn_feat_j
                )

                # Set a threshold to include edges with significant weight
                if weight > 0.001:  # Adjust threshold as needed
                    edge_index_dynamic.append([i, j])
                    edge_weight_dynamic.append(weight)

    # Convert edge lists to tensors
    if len(edge_index_dynamic) > 0:
        edge_index_dynamic = torch.tensor(edge_index_dynamic, dtype=torch.long).t().contiguous()
        edge_weight_dynamic = torch.tensor(edge_weight_dynamic, dtype=torch.float)
    else:
        edge_index_dynamic = torch.empty((2, 0), dtype=torch.long)
        edge_weight_dynamic = torch.empty((0,), dtype=torch.float)

    # Create a Data object for the current snapshot with dynamic edges
    data_dynamic = Data(
        x=node_features,              # Node features
        edge_index=edge_index_dynamic,    # Dynamic edge indices
        edge_attr=edge_weight_dynamic    # Dynamic edge weights
    )
    data_dynamic.date = current_week    # Add the week date as metadata
    dynamic_graph_snapshots.append(data_dynamic)

# Verify the number of graph snapshots
print(f"Number of static graph snapshots: {len(static_graph_snapshots)}")
print(f"Number of dynamic graph snapshots: {len(dynamic_graph_snapshots)}")
print("First static graph snapshot:")
print(static_graph_snapshots[0])
print("First dynamic graph snapshot:")
print(dynamic_graph_snapshots[0])

static_graph_snapshots

def visualize_static_graph_snapshot_plotly(graph, state_info, feature='hospitalization'):
    """
    Visualize a static graph snapshot using Plotly.
    
    Parameters:
    - graph: torch_geometric.data.Data object representing the graph.
    - state_info: DataFrame containing state information.
    - feature: Node feature to visualize (default: 'hospitalization').
    """
    G = to_networkx(graph, to_undirected=True)

    # Add node attributes
    for node in G.nodes():
        G.nodes[node][feature] = graph.x[node, FEATURE_COLUMNS.index(feature)].item()

    # Extract positions
    pos = {node: (state_info.loc[node, 'lng'], state_info.loc[node, 'lat']) for node in G.nodes()}

    # Create edge traces
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='#888'),
                hoverinfo='none'
            )
        )

    # Create node trace
    node_x = []
    node_y = []
    node_color = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(G.nodes[node][feature])
        node_text.append(state_info.loc[node, 'state'])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="bottom center",
        marker=dict(
            size=10,
            color=node_color,
            colorscale='Reds',
            colorbar=dict(title=feature),
            line_width=2
        )
    )

    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title=f"Static Graph Snapshot for Week Ending {graph.date.date()}",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper") ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )

    fig.show()

# Example: Visualize the first static graph snapshot with Plotly
visualize_static_graph_snapshot_plotly(static_graph_snapshots[0], state_info, feature='hospitalization')


def visualize_dynamic_graph_snapshot_plotly(graph, state_info, feature='hospitalization'):
    """
    Visualize a dynamic graph snapshot using Plotly.
    
    Parameters:
    - graph: torch_geometric.data.Data object representing the graph.
    - state_info: DataFrame containing state information.
    - feature: Node feature to visualize (default: 'hospitalization').
    """
    G = to_networkx(graph, to_undirected=True)

    # Add node attributes
    for node in G.nodes():
        G.nodes[node][feature] = graph.x[node, FEATURE_COLUMNS.index(feature)].item()

    # Extract positions
    pos = {node: (state_info.loc[node, 'lng'], state_info.loc[node, 'lat']) for node in G.nodes()}

    # Create edge traces
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=1, color='#888'),
                hoverinfo='none'
            )
        )

    # Create node trace
    node_x = []
    node_y = []
    node_color = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_color.append(G.nodes[node][feature])
        node_text.append(state_info.loc[node, 'state'])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="bottom center",
        marker=dict(
            size=10,
            color=node_color,
            colorscale='Reds',
            colorbar=dict(title=feature),
            line_width=2
        )
    )

    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title=f"Dynamic Graph Snapshot for Week Ending {graph.date.date()}",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper") ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                   )

    fig.show()

# Example: Visualize the first dynamic graph snapshot with Plotly
visualize_dynamic_graph_snapshot_plotly(dynamic_graph_snapshots[0], state_info, feature='hospitalization')


