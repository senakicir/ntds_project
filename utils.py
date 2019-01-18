import pandas as pd
import numpy as np
import ast
from scipy.spatial.distance import pdist, squareform
import pdb
from sklearn.decomposition import PCA
import os

SEED = 0
np.random.seed(SEED)

## Calculates the first PCA_dim dimension of the PCA
def generate_PCA_features(features, PCA_dim):
    pca = PCA(n_components=PCA_dim, svd_solver='arpack')
    return pca.fit_transform(features)

## Normalizes each columns of the features
def normalize_feat(features):
    mean_feat = np.mean(features, axis=0)
    std_feat = np.std(features, axis=0)
    normalized_feat = (features-mean_feat)/std_feat
    return normalized_feat

## Removes all disconnected nodes
def remove_disconnected_nodes(adjacency, features, labels,indx_train, indx_test):
    connected_nodes_ind = (np.sum(adjacency, axis=0) != 0)
    new_adj = adjacency[connected_nodes_ind, :][:, connected_nodes_ind]
    new_feat = features[connected_nodes_ind, :]
    temp = np.array([0]*len(labels))
    labels = labels[connected_nodes_ind]
    temp[indx_train] = 1
    temp[indx_test] = -1
    temp = temp[connected_nodes_ind]
    indx_train = np.argwhere(temp == 1).squeeze()
    indx_test =  np.argwhere(temp == -1).squeeze()
    return new_adj, new_feat, labels, indx_train, indx_test

## Generate prefix for the names of the saved files based on the different arguments
def form_file_names(use_PCA, PCA_dim, use_eigenmaps, rem_disconnected, dataset_size, threshold,use_mlp, prefix):
    if not os.path.exists("models"): os.makedirs("models")
    if not os.path.exists("visualizations"): os.makedirs("visualizations")
    if not os.path.exists("dataset_saved_numpy"): os.makedirs("dataset_saved_numpy")
    if prefix =="":
        name = prefix
    else:
         name = prefix + "_"
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
