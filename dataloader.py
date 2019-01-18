import pandas as pd
import numpy as np
import ast
from scipy.spatial.distance import pdist, squareform
from utils import remove_disconnected_nodes, generate_PCA_features, normalize_feat, uniform_random_subsample, form_file_names, SEED
from visualization import plot_gt_labels
from sklearn.manifold import spectral_embedding
import pdb
import pygsp as pg
import os
from mlp import MLP as MLP_NN

def load(filename):
    if 'features' in filename:
        return pd.read_csv(filename, index_col=0, header=[0, 1, 2])

    if 'echonest' in filename:
        return pd.read_csv(filename, index_col=0, header=[0, 1, 2])

    if 'genres' in filename:
        return pd.read_csv(filename, index_col=0)

    if 'tracks' in filename:
        tracks = pd.read_csv(filename, index_col=0, header=[0, 1])

        COLUMNS = [('track', 'tags'), ('album', 'tags'), ('artist', 'tags'),
                   ('track', 'genres'), ('track', 'genres_all')]
        for column in COLUMNS:
            tracks[column] = tracks[column].map(ast.literal_eval)

        COLUMNS = [('track', 'date_created'), ('track', 'date_recorded'),
                   ('album', 'date_created'), ('album', 'date_released'),
                   ('artist', 'date_created'), ('artist', 'active_year_begin'),
                   ('artist', 'active_year_end')]
        for column in COLUMNS:
            tracks[column] = pd.to_datetime(tracks[column])

        SUBSETS = ('small', 'medium', 'large')
        tracks['set', 'subset'] = tracks['set', 'subset'].astype(
                pd.api.types.CategoricalDtype(SUBSETS, ordered=True))

        SPLITS = ('training', 'test', 'validation')
        tracks['set', 'split'] = tracks['set', 'split'].astype(
                pd.api.types.CategoricalDtype(SPLITS, ordered=True))

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype(pd.api.types.CategoricalDtype())

    return tracks

def csv_loader():
    tracks = load('dataset/tracks.csv') #Read tracks.csv
    features = load('dataset/features.csv') # Read features.csv

    return tracks, features

def select_features(tracks, features, use_features = ['mfcc'], dataset_size = None, genres = None, num_classes=None):
    if not dataset_size:
        dataset_size = "small"

    subset_tracks = tracks[tracks['set', 'subset'] == dataset_size]

    if genres is None:
        genres = list(subset_tracks.track.genre_top.dropna().unique())
        if not(num_classes is None):
            genre_count = np.zeros([len(genres,)])
            for ind, genre in enumerate(genres):
                genre_count[ind]=(np.sum(subset_tracks['track', 'genre_top'] == genre))
                ind_sort = np.argsort(genre_count)
                genres_to_include = ind_sort[-num_classes:]
            our_genres = []
            for ind in genres_to_include:
                our_genres.append(genres[ind])
            genres = our_genres


    subset = subset_tracks.loc[subset_tracks['track', 'genre_top'].isin(genres)]
    # Takes a subset of features based on the tracks found in the variable subset
    subset_features = features[features.index.isin(subset.index)].loc[:,use_features].values

    release_dates = subset['album', 'date_released'].values

    #save labels
    dict_genres = {}
    for ind in range(0, len(genres)):
        dict_genres[genres[ind]] = int(ind)
    bool_train = np.logical_or(subset['set', 'split'] == "training", subset['set', 'split'] == "validation")
    bool_test = subset['set', 'split'] == "test"
    indx_train = np.argwhere(bool_train == True).squeeze()
    indx_test = np.argwhere(bool_test == True).squeeze()

    genres_gt = np.zeros([subset_features.shape[0]],dtype=np.int8)-1
    for ind in range(0, len(genres)):
        temp = (subset['track', 'genre_top'] == genres[ind]).to_frame().values.squeeze()
        genres_gt[temp] = dict_genres[genres[ind]]

    return subset_features, genres_gt, genres, dict_genres,indx_train,indx_test, release_dates


def form_adjacency(features, labels, genres, rem_disconnected, indx_train=0, indx_test=0, threshold = 0.66, metric ='correlation'):
    distances = pdist(features, metric=metric) # Use cosine equation to calculate distance
    kernel_width = distances.mean()
    weights = np.exp(-distances**2 / kernel_width**2) # Use Gaussian function to calculate weights
    weights[weights < threshold] = 0
    adjacency = squareform(weights)
    if rem_disconnected:
        adjacency, features, labels,indx_train,indx_test= remove_disconnected_nodes(adjacency, features, labels,indx_train,indx_test)
    num_of_disconnected_nodes = np.sum(np.sum(adjacency, axis=0) == 0)
    assert num_of_disconnected_nodes == 0
    return adjacency, features, labels, genres,indx_train,indx_test

def save_features_labels_adjacency(use_PCA = True, PCA_dim = 10, use_eigenmaps = False, rem_disconnected = True, threshold = 0.66, metric = "correlation",use_features = ['mfcc'], dataset_size = 'small',genres=None,num_classes=None, return_features=False,plot_graph=False, train=True,use_mlp=False,use_cpu=False,prefix=""):
    tracks, features = csv_loader()
    feature_values, genres_gt, genres_classes, dict_genres,indx_train,indx_test, release_dates = select_features(tracks, features, use_features = use_features, dataset_size = dataset_size, genres =genres, num_classes=num_classes)

    name = form_file_names(use_PCA, PCA_dim, use_eigenmaps, rem_disconnected, dataset_size, threshold,use_mlp,prefix)

    if (use_PCA):
        feature_values = normalize_feat(feature_values)
        feature_values = generate_PCA_features(feature_values, PCA_dim)
    if use_mlp:
        mlp_name = form_file_names(use_PCA,PCA_dim, use_eigenmaps, rem_disconnected, dataset_size, threshold,not use_mlp,prefix)
        mlp_nn = MLP_NN(hidden_size=100, features=feature_values, labels=genres_gt,num_epoch=10,batch_size=100,num_classes=len(genres_classes), save_path=mlp_name,cuda=use_cpu)
        feature_values = mlp_nn.get_rep(feature_values)
    adjacency, feature_values, genres_gt, _,indx_train,indx_test  = form_adjacency(feature_values, genres_gt, genres_classes, rem_disconnected, indx_train=indx_train, indx_test=indx_test, threshold = threshold, metric = metric)
    if (use_eigenmaps):
        feature_values = spectral_embedding(adjacency,n_components=10, eigen_solver=None,random_state=SEED, eigen_tol=0.0,norm_laplacian=True)

    np.save("dataset_saved_numpy/"+ name + "labels.npy", genres_gt)
    np.save("dataset_saved_numpy/"+ name + "adjacency.npy", adjacency)
    np.save("dataset_saved_numpy/"+ name + "features.npy", feature_values)
    np.save("dataset_saved_numpy/" + name + "genres_classes.npy", genres_classes)
    np.save("dataset_saved_numpy/" + name + "indx_train.npy", indx_train)
    np.save("dataset_saved_numpy/" + name + "indx_test.npy", indx_test)
    np.save("dataset_saved_numpy/dict_genres.npy", dict_genres)
    np.save("dataset_saved_numpy/" + name + "release_dates.npy", release_dates)
    print("Dataset of size {}. Features,labels,and genres saved using prefix: {}".format(adjacency.shape[0], name[:len(name)-1]))

    if (return_features):
        pygsp_graph = None
        if plot_graph:
            subsampled_adjacency, genres_gt = uniform_random_subsample(adjacency, genres_gt)
            pygsp_graph = pg.graphs.Graph(subsampled_adjacency, lap_type = 'normalized')
            pygsp_graph.set_coordinates('spring') #for visualization
            plot_gt_labels(pygsp_graph, genres_gt, name)
        return feature_values, genres_gt, genres_classes, adjacency,indx_train,indx_test, pygsp_graph, release_dates
    return name

def load_features_labels_adjacency(name, train, plot_graph=False):
    assert os.path.exists("dataset_saved_numpy/" + name + "features.npy")
    assert os.path.exists("dataset_saved_numpy/" + name + "adjacency.npy")
    assert os.path.exists("dataset_saved_numpy/" + name + "labels.npy")
    assert os.path.exists("dataset_saved_numpy/dict_genres.npy")
    assert os.path.exists("dataset_saved_numpy/" + name + "genres_classes.npy")
    assert os.path.exists("dataset_saved_numpy/" + name + "release_dates.npy")

    features = np.load("dataset_saved_numpy/" + name + "features.npy")
    adjacency =  np.load("dataset_saved_numpy/" + name + "adjacency.npy")
    labels = np.load("dataset_saved_numpy/" + name  + "labels.npy")
    dict_genres = np.load("dataset_saved_numpy/dict_genres.npy")
    genres_classes = np.load("dataset_saved_numpy/" + name + "genres_classes.npy")
    release_dates = np.load("dataset_saved_numpy/" + name + "release_dates.npy")
    indx_train = np.load("dataset_saved_numpy/" + name + "indx_train.npy")
    indx_test = np.load("dataset_saved_numpy/" + name + "indx_test.npy")

    pygsp_graph = None
    if plot_graph:
        subsampled_adjacency, genres_gt = uniform_random_subsample(adjacency, labels)
        pygsp_graph = pg.graphs.Graph(subsampled_adjacency, lap_type = 'normalized')
        pygsp_graph.set_coordinates('spring') #for visualization
        plot_gt_labels(pygsp_graph, genres_gt, name)
    return features, labels, genres_classes, adjacency,indx_train,indx_test,  pygsp_graph, release_dates
