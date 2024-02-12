#!/usr/bin/env python3
#
# Create Hamiltonian instances.
# It may not exactly reproduce the instances in the GitHub repo because
# the order of graph.edges() may change between networkx versions.
# To reproduce results in the paper, please directly use the instances in
# the GitHub repo.

import os

import networkx as nx
import numpy as np
from scipy.sparse import save_npz, triu

num_instances = 10
seed = 0

np.random.seed(seed)
Ls = [2, 4, 8, 16, 32, 64, 12, 24]
for L in Ls:
    # 2D Edwards-Anderson model with plus-minus one interactions
    for i in range(num_instances):
        graph = nx.grid_graph((L, L), periodic=True)
        for u, v in graph.edges():
            graph[u][v]["weight"] = np.random.randint(0, 2) * 2 - 1

        adj_matrix = nx.adjacency_matrix(graph)
        adj_matrix = triu(adj_matrix, k=1, format="csr")

        filename = f"./instances/2D_EA_PM/2D_EA_PM_L{L}_{i}.npz"
        print(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_npz(filename, adj_matrix)

    if L >= 32:
        continue

    # 3D Edwards-Anderson model with plus-minus one interactions
    for i in range(num_instances):
        graph = nx.grid_graph((L, L, L), periodic=True)
        for u, v in graph.edges():
            graph[u][v]["weight"] = np.random.randint(0, 2) * 2 - 1

        adj_matrix = nx.adjacency_matrix(graph)
        adj_matrix = triu(adj_matrix, k=1, format="csr")

        filename = f"./instances/3D_EA_PM/3D_EA_PM_L{L}_{i}.npz"
        print(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_npz(filename, adj_matrix)

# Random regular graph with plus-minus one interactions
d = 3
Ns = [32, 64, 128, 256, 512, 1024]
for N in Ns:
    for i in range(10):
        np.random.seed(i)
        graph = nx.random_regular_graph(d=d, n=N, seed=i)
        weights = np.random.randint(2, size=len(graph.edges)) * 2 - 1
        for (u, v), w in zip(graph.edges(), weights):
            graph[u][v]["weight"] = w

        adj_matrix = nx.adjacency_matrix(graph)
        adj_matrix = triu(adj_matrix, k=1, format="csr")

        filename = f"./instances/RRG_PM/RRG_PM_d{d}_N{N}_{i}.npz"
        print(filename)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        save_npz(filename, adj_matrix)
