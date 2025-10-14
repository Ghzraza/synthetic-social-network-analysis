# clustering.py
# Author: Ghazal Raza
# Purpose: Functions to perform community detection and clustering on networks

import networkx as nx
from sklearn.cluster import SpectralClustering
import numpy as np

def cluster_network(G, n_clusters=2):
    """
    Perform spectral clustering on a network.

    Args:
        G (networkx.Graph): Input network.
        n_clusters (int): Number of clusters/communities.

    Returns:
        labels (ndarray): Cluster labels for each node.
    """
    adj = nx.to_numpy_array(G)
    clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
    labels = clustering.fit_predict(adj)
    return labels
