import pandas as pd
import numpy as np
import ast
from scipy.spatial.distance import pdist, squareform
import pdb
from sklearn.decomposition import PCA
import os

SEED = 0
np.random.seed(SEED)

def generate_PCA_features(features, PCA_dim):
    pca = PCA(n_components=PCA_dim, svd_solver='arpack')
    return pca.fit_transform(features)

def normalize_feat(features):
    mean_feat = np.mean(features, axis=0)
    std_feat = np.std(features, axis=0)
    normalized_feat = (features-mean_feat)/std_feat
    return normalized_feat

def remove_disconnected_nodes(adjacency, features, labels):
    connected_nodes_ind = (np.sum(adjacency, axis=0) != 0)
    new_adj = adjacency[connected_nodes_ind, :][:, connected_nodes_ind]
    new_feat = features[connected_nodes_ind, :]
    labels = labels[connected_nodes_ind]
    return new_adj, new_feat, labels

def uniform_random_subsample(adjacency, genres_gt, subsampling_percentage=0.10):
    n_nodes = adjacency.shape[0]
    shuffled_ind = np.random.permutation(n_nodes)
    shuffled_ind_subsampled = shuffled_ind[0:int(n_nodes*subsampling_percentage)]
    adjacency = adjacency[:, shuffled_ind_subsampled][shuffled_ind_subsampled, :]
    genres_gt = genres_gt[shuffled_ind_subsampled]
    return adjacency, genres_gt

def form_file_names(use_PCA, PCA_dim, use_eigenmaps, rem_disconnected, dataset_size, threshold,use_mlp):
    if not os.path.exists("models"): os.makedirs("models")
    if not os.path.exists("visualizations"): os.makedirs("visualizations")
    if not os.path.exists("dataset_saved_numpy"): os.makedirs("dataset_saved_numpy")

    name = ""
    if use_PCA:
        name += "PCA_+dim_"+str(PCA_dim)+"_"

    if rem_disconnected:
        name += "rem_disconnected_"

    if use_mlp:
        name += "useMLP_"

    if use_eigenmaps:
        name += "use_eigmaps_"

    name += dataset_size + "_"
    name += "thr" + str(threshold) + "_"
    return name
