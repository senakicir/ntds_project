import pandas as pd
import numpy as np
import ast
import scipy.sparse as sp
from scipy.spatial.distance import pdist, squareform
import pdb
import pygsp as pg
from scipy import sparse
import scipy.sparse.linalg
from matplotlib import pyplot as plt
from pyunlocbox import functions, solvers
import networkx as nx
from error import error_func

class Our_Graph():
    def __init__(self, adjacency):
        self.adjacency = adjacency
        self.n_nodes = len(adjacency)
        self.n_edges = np.count_nonzero(adjacency)//2
        self.D = np.diag(adjacency.sum(axis=1))
        self.D_norm = np.power(np.linalg.inv(self.D),0.5)
        self.laplacian_combinatorial = self.D - self.adjacency # combinatorial laplacian
        self.laplacian_normalized = self.D_norm @ self.laplacian_combinatorial @ self.D_norm
        #self.compute_gradient(normalized=False)
        self.compute_fourier_basis(normalized=False)

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

    def get_laplacian_eigenmaps(self, new_dim=10, use_normalized=True):
        if use_normalized:
            laplacian = self.laplacian_normalized
        else:
            laplacian = self.laplacian_combinatorial
        eigenvalues, eigenvectors = sparse.linalg.eigsh(laplacian, which='SM', k=new_dim + 1)
        sort_ind = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_ind]
        eigenvectors = eigenvectors[:,sort_ind]
        embeddings = eigenvectors[:, 1:new_dim + 1]
        if use_normalized:
            return (embeddings.T * np.diag(self.D_norm)).T
        return embeddings

    def spectral_clustering(self, test_pos, labels_bin):
        fiedler_vector_c = self.get_laplacian_eigenmaps(new_dim=1, use_normalized=False).squeeze()
        fiedler_vector_n = self.get_laplacian_eigenmaps(new_dim=1, use_normalized=True).squeeze()

        # We do clustering according to the fiedler vector
        clusters_c = np.zeros([self.n_nodes, ])
        clusters_c[fiedler_vector_c > 0] = 1
        clusters_c[fiedler_vector_c <= 0] = -1

        clusters_n = np.zeros([self.n_nodes, ])
        clusters_n[fiedler_vector_n > 0] = 1
        clusters_n[fiedler_vector_n <= 0] = -1

        clusters_c = clusters_c[test_pos]
        clusters_n = clusters_n[test_pos]

        print('Percentage Error with fiedler vector combinatorial: {:.2f}'.format(error_func(labels_bin, clusters_c)))
        print('Percentage Error with fiedler vector normalized: {:.2f}'.format(error_func(labels_bin, clusters_n)))

    def graph_pnorm_interpolation(self, gradient, P, w, labels_bin, x0=None, p=1., **kwargs):
        r"""
        Solve an interpolation problem via gradient p-norm minimization.

        A signal :math:`x` is estimated from its measurements :math:`y = A(x)` by solving
        :math:`\text{arg}\underset{z \in \mathbb{R}^n}{\min}
        \| \nabla_G z \|_p^p \text{ subject to } Az = y` 
        via a primal-dual, forward-backward-forward algorithm.

        Parameters
        ----------
        gradient : array_like
            A matrix representing the graph gradient operator
        P : callable
            Orthogonal projection operator mapping points in :math:`z \in \mathbb{R}^n` 
            onto the set satisfying :math:`A P(z) = A z`.
        x0 : array_like, optional
            Initial point of the iteration. Must be of dimension n.
            (Default is `numpy.random.randn(n)`)
        p : {1., 2.}
        labels_bin : array_like
            A vector that holds the binary labels.
        kwargs :
            Additional solver parameters, such as maximum number of iterations
            (maxit), relative tolerance on the objective (rtol), and verbosity
            level (verbosity). See :func:`pyunlocbox.solvers.solve` for the full
            list of options.

        Returns
        -------
        x : array_like
            The solution to the optimization problem.

        """

        grad = lambda z: gradient.dot(z)
        div = lambda z: gradient.transpose().dot(z)

        # Indicator function of the set satisfying :math:`y = A(z)`
        f = functions.func()
        f._eval = lambda z: 0
        f._prox = lambda z, gamma: P(z, w, labels_bin)

        # :math:`\ell_1` norm of the dual variable :math:`d = \nabla_G z`
        g = functions.func()
        g._eval = lambda z: np.sum(np.abs(grad(z)))
        g._prox = lambda d, gamma: functions._soft_threshold(d, gamma)

        # :math:`\ell_2` norm of the gradient (for the smooth case)
        h = functions.norm_l2(A=grad, At=div)

        stepsize = (0.9 / (1. + scipy.sparse.linalg.norm(gradient, ord='fro'))) ** p

        solver = solvers.mlfbf(L=grad, Lt=div, step=stepsize)

        if p == 1.:
            problem = solvers.solve([f, g, functions.dummy()], x0=x0, solver=solver, **kwargs)
            return problem['sol']
        if p == 2.:
            problem = solvers.solve([f, functions.dummy(), h], x0=x0, solver=solver, **kwargs)
            return problem['sol']
        else:
            return x0

    def P(self, a, w, labels_bin):
        mask_pos = np.where(w == 1)
        b = a.copy()
        b[mask_pos] = labels_bin[mask_pos]
        return b

    def transductive_learning(self, w, thresholds, n_trials, labels_bin, test_ind, p, **kwargs):
        for _, threshold in enumerate(thresholds):
            # Simulate n_trials times
            for _ in range(n_trials):
                # Subsample randomly
                subsampled = labels_bin * w
                # Solve p-norm interpolation
                sol = self.graph_pnorm_interpolation(sparse.csr_matrix(self.S).T, self.P, w, labels_bin,
                                                x0=subsampled.copy(), p=p, **kwargs)
                # Threshold to -1 and 1
                sol_bin = (sol > threshold).astype(int)
                sol_bin[sol_bin == 0] = -1
                pdb.set_trace()
                print('Percentage Error with threshold: {:.2f}'.format(error_func(labels_bin[test_ind], sol_bin[test_ind])))

    