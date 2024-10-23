import networkx as nx
import numpy as np

# Create an undirected graph
G = nx.Graph()
G.add_edges_from(
    [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("C", "G")]
)

# visualize the graph
nx.draw(G, with_labels=True)

# Create a directed graph
DG = nx.DiGraph()
DG.add_edges_from(
    [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("C", "G")]
)

# visualize the directed graph
nx.draw(DG, with_labels=True)

import matplotlib.pyplot as plt

# Create a weighted graph
WG = nx.Graph()
WG.add_edges_from(
    [
        ("A", "B", {"weight": 10}),
        ("A", "C", {"weight": 20}),
        ("B", "D", {"weight": 30}),
        ("B", "E", {"weight": 40}),
        ("C", "F", {"weight": 50}),
        ("C", "G", {"weight": 60}),
    ]
)

# Get edge labels
labels = nx.get_edge_attributes(WG, "weight")

# Visualize the weighted graph
pos = nx.spring_layout(WG)
nx.draw(WG, pos, with_labels=True)
nx.draw_networkx_edge_labels(WG, pos, edge_labels=labels)

plt.show()

# Print degree of node 'A' in the undirected graph
print(f"deg(A) = {G.degree['A']}")

# Print in-degree and out-degree of node 'A' in the directed graph
print(f"deg^-(A) = {DG.in_degree['A']}")
print(f"deg^+(A) = {DG.out_degree['A']}")

# Calculate and print centrality measures for the undirected graph
print(f"Degree centrality = {nx.degree_centrality(G)}")
print(f"Closeness centrality = {nx.closeness_centrality(G)}")
print(f"Betweenness centrality = {nx.betweenness_centrality(G)}")

# Convert adjacency matrix to a NetworkX graph
adj = [
    [0, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 1],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0],
]

# Convert edge list to a NetworkX graph
edge_list = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
G_from_edge_list = nx.Graph(edge_list)
print(f"Graph from edge list: {G_from_edge_list.edges()}")

# Convert adjacency list to a NetworkX graph
adj_list = {0: [1, 2], 1: [0, 3, 4], 2: [0, 5, 6], 3: [1], 4: [1], 5: [2], 6: [2]}
G_from_adj_list = nx.Graph(adj_list)
print(f"Graph from adjacency list: {G_from_adj_list.edges()}")
