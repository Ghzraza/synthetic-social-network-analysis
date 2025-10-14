# network_evolution.py
# Author: Ghazal Raza
# Purpose: Functions to simulate temporal evolution of networks

import numpy as np
import networkx as nx

def evolve_network(G, steps=10, p_add=0.1, p_remove=0.05):
    """
    Simulate temporal evolution of a network.

    Args:
        G (networkx.Graph): Initial network.
        steps (int): Number of time steps to evolve.
        p_add (float): Probability to add edges.
        p_remove (float): Probability to remove edges.

    Returns:
        evolution (list): List of networkx.Graph at each time step.
    """
    evolution = []
    G_curr = G.copy()
    for t in range(steps):
        G_next = G_curr.copy()
        # Add edges randomly
        for i in G_curr.nodes():
            for j in G_curr.nodes():
                if i != j and not G_curr.has_edge(i,j) and np.random.rand() < p_add:
                    G_next.add_edge(i,j)
        # Remove edges randomly
        for i,j in list(G_curr.edges()):
            if np.random.rand() < p_remove:
                G_next.remove_edge(i,j)
        evolution.append(G_next)
        G_curr = G_next
    return evolution
