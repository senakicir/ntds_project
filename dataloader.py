import pandas as pd
import numpy as np
import ast
from scipy.spatial.distance import pdist, squareform
from utils import remove_disconnected_nodes, generate_PCA_features, normalize_feat, uniform_random_subsample, form_file_names
from visualization import plot_gt_labels
import pdb
import pygsp as pg
import os

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

        COLUMNS = [('track', 'genre_top'), ('track', 'license'),
                   ('album', 'type'), ('album', 'information'),
                   ('artist', 'bio')]
        for column in COLUMNS:
            tracks[column] = tracks[column].astype(pd.api.types.CategoricalDtype())

    return tracks

def csv_loader():
    tracks = load('dataset/tracks.csv') #Read tracks.csv
    features = load('dataset/features.csv') # Read features.csv

    # MAYBE WE WONT NEED THESE BUT I KEPT THEM ANYWAY

    # Merges feature dataframe with genre_top column in tracks based on track_id
    #merged_full = pd.merge(features, tracks['track', 'genre_top'].to_frame(), on="track_id")
    # Separates the 16 top genres into columns with binary encoding
    #merged_full_binary = merged_full.join(pd.get_dummies(merged_full.pop(('track', 'genre_top'))))

    return tracks, features

def select_features(tracks, features, use_features = ['mfcc'], dataset_size = None, genres = None, num_classes=None):
    if dataset_size:
        small = tracks[tracks['set', 'subset'] == dataset_size]
    else:
        small = tracks

    if genres is None:
        genres = list(small.track.genre_top.dropna().unique())
        if not(num_classes is None):
            genres = genres[:num_classes]

    subset = small.loc[small['track', 'genre_top'].isin(genres)]
    # Takes a subset of features based on the tracks found in the variable subset
    small_features = features[features.index.isin(subset.index)]
    #features_part_mfcc = small_features.loc[:,('mfcc', ('median', 'mean'), slice('01','12'))]
    #features_part_chroma = small_features.loc[:,('chroma_cens', ('median', 'mean'), slice('01','05'))] # Take chroma column as features
    #features_part = features_part_mfcc.join(features_part_chroma)
    features_part = small_features.loc[:,use_features].values
    release_dates = subset['album', 'date_released'].values

    #save labels
    genres_gt = np.zeros([features_part.shape[0]],dtype=np.int8)-1
    genres_gt_onehot = np.zeros([features_part.shape[0],len(genres)],dtype=np.int8)
    dict_genres = {}
    for ind in range(0, len(genres)):
        temp = (subset['track', 'genre_top'] == genres[ind]).to_frame().values.squeeze()
        genres_gt[temp] = int(ind)
        dict_genres[genres[ind]] = int(ind)
    if -1 in genres_gt:
        raise ValueError('Not all tracks were labeled')
    genres_gt_onehot[np.arange(features_part.shape[0]),genres_gt] = 1
    return features_part, genres_gt, genres_gt_onehot,genres, dict_genres, release_dates


def form_adjacency(features, labels, labels_one_hot, rem_disconnected, threshold = 0.66, metric ='correlation'):
    distances = pdist(features, metric=metric) # Use cosine equation to calculate distance
    kernel_width = distances.mean()
    weights = np.exp(-distances**2 / kernel_width**2) # Use Gaussian function to calculate weights
    weights[weights < threshold] = 0
    adjacency = squareform(weights)
    if rem_disconnected:
        adjacency, features, labels, labels_one_hot = remove_disconnected_nodes(adjacency, features, labels, labels_one_hot)
    num_of_disconnected_nodes = np.sum(np.sum(adjacency, axis=0) == 0)
    assert num_of_disconnected_nodes == 0
    return adjacency, features, labels, labels_one_hot

def save_features_labels_adjacency(normalize_features = True, use_PCA = True, rem_disconnected = True, threshold = 0.66, metric = "correlation",use_features = ['mfcc'], dataset_size = 'small',genres=None,num_classes=None, return_features=False,plot_graph=False):
    tracks, features = csv_loader()
    #feature_values, genres_gt,genres_gt_onehot,genres_classes, dict_genres = select_features(tracks, features, use_features = ['mfcc'], dataset_size = 'small', genres = ['Hip-Hop', 'Rock'], num_classes=num_classes)
    feature_values, genres_gt, genres_gt_onehot, genres_classes, dict_genres, release_dates = select_features(tracks, features, use_features = use_features, dataset_size = dataset_size,genres =genres, num_classes=num_classes)

    name = form_file_names(normalize_features, use_PCA, rem_disconnected, dataset_size, threshold)
    if (normalize_features):
        feature_values = normalize_feat(feature_values)
    if (use_PCA):
        feature_values = generate_PCA_features(feature_values)

    adjacency, feature_values, genres_gt, genres_gt_onehot  = form_adjacency(feature_values, genres_gt, genres_gt_onehot, rem_disconnected,  threshold = threshold, metric = metric)

    np.save("dataset_saved_numpy/labels.npy", genres_gt)
    np.save("dataset_saved_numpy/labels_onehot.npy",genres_gt_onehot)
    np.save("dataset_saved_numpy/"+ name + "adjacency.npy", adjacency)
    np.save("dataset_saved_numpy/"+ name + "features.npy", feature_values)
    np.save("dataset_saved_numpy/genres_classes.npy", genres_classes)
    np.save("dataset_saved_numpy/dict_genres.npy", dict_genres)
    np.save("dataset_saved_numpy/release_dates.npy", release_dates)
    print("Dataset of size {}. Features,labels,and genres saved using prefix: {}".format(adjacency.shape[0], name[:len(name)-1]))

    if (return_features):
        pygsp_graph = None
        if plot_graph:
            subsampled_adjacency, genres_gt, _ = uniform_random_subsample(adjacency, genres_gt, genres_gt_onehot)
            pygsp_graph = pg.graphs.Graph(subsampled_adjacency, lap_type = 'normalized')
            pygsp_graph.set_coordinates('spring') #for visualization
            plot_gt_labels(pygsp_graph, genres_gt, name)
        return feature_values, genres_gt, genres_gt_onehot, genres_classes, adjacency, pygsp_graph, release_dates
    return name

def load_features_labels_adjacency(name, plot_graph=False):
    assert os.path.exists("dataset_saved_numpy/" + name + "features.npy")
    assert os.path.exists("dataset_saved_numpy/" + name + "adjacency.npy")
    assert os.path.exists("dataset_saved_numpy/labels.npy")
    assert os.path.exists("dataset_saved_numpy/labels_onehot.npy")
    assert os.path.exists("dataset_saved_numpy/dict_genres.npy")
    assert os.path.exists("dataset_saved_numpy/genres_classes.npy")
    assert os.path.exists("dataset_saved_numpy/release_dates.npy")

    features = np.load("dataset_saved_numpy/" + name + "features.npy")
    adjacency =  np.load("dataset_saved_numpy/" + name + "adjacency.npy")
    labels = np.load("dataset_saved_numpy/labels.npy")
    labels_onehot = np.load("dataset_saved_numpy/labels_onehot.npy")
    dict_genres = np.load("dataset_saved_numpy/dict_genres.npy")
    genres_classes = np.load("dataset_saved_numpy/genres_classes.npy")
    release_dates = np.load("dataset_saved_numpy/release_dates.npy")

    pygsp_graph = None
    if plot_graph:
        subsampled_adjacency, genres_gt, _ = uniform_random_subsample(adjacency, labels, labels_onehot)
        pygsp_graph = pg.graphs.Graph(subsampled_adjacency, lap_type = 'normalized')
        pygsp_graph.set_coordinates('spring') #for visualization
        plot_gt_labels(pygsp_graph, genres_gt, name)
    return features, labels, labels_onehot, genres_classes, adjacency,  pygsp_graph, release_dates
