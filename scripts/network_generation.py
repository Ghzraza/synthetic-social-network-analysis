# network_generation.py
# Author: Ghazal Raza
# Purpose: Functions to generate synthetic networks for social/neural interaction modeling

import networkx as nx
import random
import numpy as np

def generate_synthetic_network(num_nodes=20, p_edge=0.2, seed=42):
    """
    Generate a synthetic network with node types.

    Args:
        num_nodes (int): Number of nodes.
        p_edge (float): Probability of edge creation.
        seed (int): Random seed for reproducibility.

    Returns:
        G (networkx.Graph): Generated network.
    """
    np.random.seed(seed)
    G = nx.erdos_renyi_graph(n=num_nodes, p=p_edge, seed=seed)
    for node in G.nodes():
        G.nodes[node]['type'] = random.choice(['agent_A','agent_B'])  # analogous to neuron types or social roles
    return G
