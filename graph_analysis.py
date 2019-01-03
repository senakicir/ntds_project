import pandas as pd
import numpy as np
import ast
from scipy.spatial.distance import pdist, squareform
import pdb
import pygsp as pg

class Graph():
    def __init__(self, adjacency):
        self.adjacency = adjacency
        self.D = np.diag(adjacency.sum(axis=1))
        self.sqrt_D = np.power(np.linalg.inv(D), 0.5)

    def form_laplacian(self):
        self.laplacian = self.D - self.adjacency  # combinatorial laplacian
        return self.laplacian

    def form_incidence(self):
        # Find incidence matrix. Since our graph is undirected, we chose to only consider the upper right
        # triangle of our adjacency matrix when finding the gradient.
        normalized_incidence = np.zeros((n_nodes, n_edges))
        edge_map = np.full((n_nodes, n_nodes), -1)
        edge_indices = np.where(adjacency > 0)
        next_edge = 0
        for i, j in (zip(edge_indices[0], edge_indices[1])):
            if edge_map[i, j] == -1:
                edge_map[i, j] = next_edge
                edge_map[j, i] = next_edge
                normalized_incidence[i, next_edge] = np.sqrt(adjacency[i, j]) * sqrt_D[i, i]
                next_edge += 1
            else:
                normalized_incidence[i, edge_map[i, j]] = -np.sqrt(adjacency[i, j]) * sqrt_D[i, i]
        self.normalized_incidence = normalized_incidence
        return self.normalized_incidence

    # # Question part
    # laplacian = sparse.csr_matrix(sqrt_D @ L @ sqrt_D)
    # gradient = sparse.csr_matrix(np.transpose(normalized_incidence))
    # labels = np.load('labels.npy')  # the labels for hip-hop and rock (-1 and 1)
    # print('Number of nodes: ', n_nodes)
    # eigenvalues, eigenvectors = np.linalg.eigh(laplacian.toarray())
    # ## Sorting Eigenvalues and EigenVectors
    # sorted_indexes = eigenvalues.argsort()
    # eigenvalues = np.array(eigenvalues[sorted_indexes])
    # eigenvectors = np.array(eigenvectors[sorted_indexes])
    #
    # e = eigenvalues  # Ordered Laplacian eigenvalues.
    # U = eigenvectors  # Ordered graph Fourier basis.
    # # We use it only for plotting!
    # graph = pg.graphs.Graph(adjacency, lap_type='normalized')
    # graph.set_coordinates('spring')
if __name__ == "__main__":
    save_adjacency_matrix()
    