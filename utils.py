import pandas as pd
import numpy as np
import ast
from scipy.spatial.distance import pdist, squareform
import pdb
from sklearn.decomposition import PCA
import os

def generate_PCA_features(features):
    pca = PCA(n_components=10, svd_solver='arpack')
    return pca.fit_transform(features)

def normalize_feat(features):
    mean_feat = np.mean(features, axis=0)
    std_feat = np.std(features, axis=0)
    normalized_feat = (features-mean_feat)/std_feat
    return normalized_feat

def remove_disconnected_nodes(adjacency, features, labels, labels_one_hot):
    connected_nodes_ind = (np.sum(adjacency, axis=0) != 0)
    new_adj = adjacency[connected_nodes_ind, :][:, connected_nodes_ind]
    new_feat = features[connected_nodes_ind, :]
    labels = labels[connected_nodes_ind]
    labels_one_hot = labels_one_hot[connected_nodes_ind,:]
    return new_adj, new_feat, labels, labels_one_hot

def uniform_random_subsample(adjacency, genres_gt, genres_gt_onehot, subsampling_percentage=0.10):
    n_nodes = adjacency.shape[0]
    shuffled_ind = np.random.permutation(n_nodes)
    shuffled_ind_subsampled = shuffled_ind[0:int(n_nodes*subsampling_percentage)]
    adjacency = adjacency[:, shuffled_ind_subsampled][shuffled_ind_subsampled, :]
    genres_gt = genres_gt[shuffled_ind_subsampled]
    genres_gt_onehot = genres_gt_onehot[shuffled_ind_subsampled, :]
    return adjacency, genres_gt, genres_gt_onehot

def form_file_names(normalize_features, use_PCA, rem_disconnected, dataset_size, threshold):
    if not os.path.exists("models"): os.makedirs("models")
    if not os.path.exists("visualizations"): os.makedirs("visualizations")
    if not os.path.exists("dataset_saved_numpy"): os.makedirs("dataset_saved_numpy")

    name = ""
    if (normalize_features):
        name += "normalized_"
    if use_PCA:
        name += "PCA_"
    if rem_disconnected:
        name += "rem_disconnected_"
    name += dataset_size + "_"
    name += "thr" + str(threshold) + "_"
    
    return name