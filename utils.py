import pandas as pd
import numpy as np
import ast
from scipy.spatial.distance import pdist, squareform
import pdb
from sklearn.decomposition import PCA

def generate_PCA_features(features):
    pca = PCA(n_components=10, svd_solver='arpack')
    return pca.fit_transform(features)

def remove_outliers(features):
    return features

def normalize_feat(features):
    mean_feat = np.mean(features, axis=0)
    std_feat = np.std(features, axis=0)
    normalized_feat = (features-mean_feat)/std_feat
    return normalized_feat
