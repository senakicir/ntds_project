import pandas as pd
import numpy as np
import ast
from scipy.spatial.distance import pdist, squareform
import pdb
import pygsp as pg

class Our_Graph():
    def __init__(self, adjacency):
        self.adjacency = adjacency
        self.n_nodes = len(adjacency)
        self.n_edges = np.count_nonzero(adjacency)//2
        self.D = np.diag(adjacency.sum(axis=1))
       # self.D_norm = np.diag(np.sum(adjacency, 1) ** (-1 / 2))
        self.D_norm = np.power(np.linalg.inv(self.D),0.5)
        self.laplacian_combinatorial = self.D - self.adjacency # combinatorial laplacian
        self.laplacian_normalized = self.D_norm @ self.laplacian_combinatorial @ self.D_norm

    def get_laplacian(self, normalized=False):
        self.laplacian = self.laplacian_combinatorial  # combinatorial laplacian
        if normalized:
            self.laplacian = self.laplacian_normalized
        return self.laplacian

    def compute_gradient(self, normalized=False):
        # Find incidence matrix. Since our graph is undirected, we chose to only consider the upper right
        # triangle of our adjacency matrix when finding the gradient.
        self.S = np.zeros((self.n_nodes, self.n_edges))
        self.S_normalized = np.zeros((self.n_nodes, self.n_edges))
        edge_idx = 0
        for i in range(self.n_nodes):
            for k in range(i):
                if self.adjacency[i, k] != 0:
                    self.S[i, edge_idx] = np.sqrt(self.adjacency[i,k])
                    self.S[k, edge_idx] = -np.sqrt(self.adjacency[i,k])
                    self.S_normalized[i, edge_idx] = np.sqrt(self.adjacency[i,k]) * self.D_norm[i,i]
                    self.S_normalized[k, edge_idx] = -np.sqrt(self.adjacency[i,k]) * self.D_norm[k,k]
                    edge_idx += 1

        assert np.allclose(self.S @ self.S.T, self.laplacian_combinatorial), "Wrong incidence matrix"
        assert np.allclose(self.S_normalized @ self.S_normalized.T, self.laplacian_normalized), "Wrong normalized incidence matrix"

        if normalized:
            return self.S
        else:
            return self.S_normalized

    def compute_fourier_basis(self, normalized=False):
        # e: Ordered Laplacian eigenvalues
        # U: Ordered graph Fourier basis
        self.e, self.U = np.linalg.eigh(self.laplacian_combinatorial)
        self.e_normalized, self.U_normalized = np.linalg.eigh(self.laplacian_normalized)
        if normalized:
            return self.e_normalized, self.U_normalized
        else:
            return self.e, self.U

    