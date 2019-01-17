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

    # MAYBE WE WONT NEED THESE BUT I KEPT THEM ANYWAY

    # Merges feature dataframe with genre_top column in tracks based on track_id
    #merged_full = pd.merge(features, tracks['track', 'genre_top'].to_frame(), on="track_id")
    # Separates the 16 top genres into columns with binary encoding
    #merged_full_binary = merged_full.join(pd.get_dummies(merged_full.pop(('track', 'genre_top'))))

    return tracks, features

def select_features(tracks, features, train, use_features = ['mfcc'], dataset_size = None, genres = None, num_classes=None):
    if not dataset_size:
        dataset_size = "small"
    if train:
        subset_tracks = tracks[np.logical_and(np.logical_or(tracks['set', 'split'] == "training", tracks['set', 'split'] == "validation"), tracks['set', 'subset'] == dataset_size)]
    else:
        subset_tracks = tracks[np.logical_and(tracks['set', 'split'] == "test", tracks['set', 'subset'] == dataset_size)]

    #subset_tracks = tracks[tracks['set', 'subset'] == dataset_size]

    if genres is None:
        genres = list(subset_tracks.track.genre_top.dropna().unique())
        if not(num_classes is None):
            genres = genres[:num_classes]

    subset = subset_tracks.loc[subset_tracks['track', 'genre_top'].isin(genres)]
    # Takes a subset of features based on the tracks found in the variable subset
    subset_features = features[features.index.isin(subset.index)]
    #features_part_mfcc = small_features.loc[:,('mfcc', ('median', 'mean'), slice('01','12'))]
    #features_part_chroma = small_features.loc[:,('chroma_cens', ('median', 'mean'), slice('01','05'))] # Take chroma column as features
    #features_part = features_part_mfcc.join(features_part_chroma)
    features_part = subset_features.loc[:,use_features].values
    release_dates = subset['album', 'date_released'].values

    #save labels
    genres_gt = np.zeros([features_part.shape[0]],dtype=np.int8)-1
    dict_genres = {}
    for ind in range(0, len(genres)):
        temp = (subset['track', 'genre_top'] == genres[ind]).to_frame().values.squeeze()
        genres_gt[temp] = int(ind)
        dict_genres[genres[ind]] = int(ind)
    if -1 in genres_gt:
        raise ValueError('Not all tracks were labeled')
    return features_part, genres_gt, genres, dict_genres, release_dates


def form_adjacency(features, labels, genres, rem_disconnected, threshold = 0.66, metric ='correlation'):
    distances = pdist(features, metric=metric) # Use cosine equation to calculate distance
    kernel_width = distances.mean()
    weights = np.exp(-distances**2 / kernel_width**2) # Use Gaussian function to calculate weights
    weights[weights < threshold] = 0
    adjacency = squareform(weights)
    if rem_disconnected:
        adjacency, features, labels = remove_disconnected_nodes(adjacency, features, labels)
        for i in range(len(genres)):
            labels_count = np.sum(labels == i)
            if labels_count == 0:
                removed_genre = genres.pop(i)
                print("Genre ", removed_genre, " was removed.")
    num_of_disconnected_nodes = np.sum(np.sum(adjacency, axis=0) == 0)
    assert num_of_disconnected_nodes == 0
    return adjacency, features, labels, genres

def save_features_labels_adjacency(normalize_features = True, use_PCA = True, rem_disconnected = True, threshold = 0.66, metric = "correlation",use_features = ['mfcc'], dataset_size = 'small',genres=None,num_classes=None, return_features=False,plot_graph=False, train=True):
    tracks, features = csv_loader()
    feature_values, genres_gt, genres_classes, dict_genres, release_dates = select_features(tracks, features, train, use_features = use_features, dataset_size = dataset_size,genres =genres, num_classes=num_classes)

    name = form_file_names(normalize_features, use_PCA, rem_disconnected, dataset_size, threshold, train)
    if (normalize_features):
        feature_values = normalize_feat(feature_values)
    if (use_PCA):
        feature_values = generate_PCA_features(feature_values)

    adjacency, feature_values, genres_gt, genres_classes  = form_adjacency(feature_values, genres_gt, genres_classes, rem_disconnected,  threshold = threshold, metric = metric)

    np.save("dataset_saved_numpy/"+ name + "labels.npy", genres_gt)
    np.save("dataset_saved_numpy/"+ name + "adjacency.npy", adjacency)
    np.save("dataset_saved_numpy/"+ name + "features.npy", feature_values)
    np.save("dataset_saved_numpy/" + name + "genres_classes.npy", genres_classes)
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
        return feature_values, genres_gt, genres_classes, adjacency, pygsp_graph, release_dates
    return name

def load_features_labels_adjacency(name, plot_graph=False):
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

    pygsp_graph = None
    if plot_graph:
        subsampled_adjacency, genres_gt, _ = uniform_random_subsample(adjacency, labels)
        pygsp_graph = pg.graphs.Graph(subsampled_adjacency, lap_type = 'normalized')
        pygsp_graph.set_coordinates('spring') #for visualization
        plot_gt_labels(pygsp_graph, genres_gt, name)
    return features, labels, genres_classes, adjacency,  pygsp_graph, release_dates
