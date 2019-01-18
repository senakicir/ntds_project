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

    # MAYBE WE WONT NEED THESE BUT I KEPT THEM ANYWAY

    # Merges feature dataframe with genre_top column in tracks based on track_id
    #merged_full = pd.merge(features, tracks['track', 'genre_top'].to_frame(), on="track_id")
    # Separates the 16 top genres into columns with binary encoding
    #merged_full_binary = merged_full.join(pd.get_dummies(merged_full.pop(('track', 'genre_top'))))

    return tracks, features

def select_features(tracks, features, use_features = ['mfcc'], dataset_size = None, genres = None, num_classes=None):
    if not dataset_size:
        dataset_size = "small"

    #train_ind = tracks[np.logical_and(tracks['set', 'subset'] == dataset_size, (np.logical_or(tracks['set', 'split'] == "training", tracks['set', 'split'] == "validation")))]
    #test_ind = tracks[np.logical_and(tracks['set', 'subset'] == dataset_size, (tracks['set', 'split'] == "test"))]
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
    subset_features = features[features.index.isin(subset.index)]
    #features_part_mfcc = small_features.loc[:,('mfcc', ('median', 'mean'), slice('01','12'))]
    #features_part_chroma = small_features.loc[:,('chroma_cens', ('median', 'mean'), slice('01','05'))] # Take chroma column as features
    #features_part = features_part_mfcc.join(features_part_chroma)

    #features_part = subset_features
    release_dates = subset['album', 'date_released'].values

    #save labels
    dict_genres = {}
    for ind in range(0, len(genres)):
        dict_genres[genres[ind]] = int(ind)
    subset_train = subset[np.logical_or(subset['set', 'split'] == "training", subset['set', 'split'] == "validation")]
    subset_test = subset[subset['set', 'split'] == "test"]

    features_part_train = subset_features[subset_features.index.isin(subset_train.index)].loc[:,use_features].values
    features_part_test = subset_features[subset_features.index.isin(subset_test.index)].loc[:,use_features].values

    genres_gt_train = np.zeros([features_part_train.shape[0]],dtype=np.int8)-1
    genres_gt_test = np.zeros([features_part_test.shape[0]],dtype=np.int8)-1

    for ind in range(0, len(genres)):
        temp = (subset_train['track', 'genre_top'] == genres[ind]).to_frame().values.squeeze()
        genres_gt_train[temp] = dict_genres[genres[ind]]
        temp = (subset_test['track', 'genre_top'] == genres[ind]).to_frame().values.squeeze()
        genres_gt_test[temp] = dict_genres[genres[ind]]

    return features_part_train, features_part_test, genres_gt_train, genres_gt_test, genres, dict_genres, release_dates


def form_adjacency(features, labels, genres, rem_disconnected, idx_tr=0, idx_test=0, threshold = 0.66, metric ='correlation'):
    distances = pdist(features, metric=metric) # Use cosine equation to calculate distance
    kernel_width = distances.mean()
    weights = np.exp(-distances**2 / kernel_width**2) # Use Gaussian function to calculate weights
    weights[weights < threshold] = 0
    adjacency = squareform(weights)
    if rem_disconnected:
        adjacency, features, labels= remove_disconnected_nodes(adjacency, features, labels)
    num_of_disconnected_nodes = np.sum(np.sum(adjacency, axis=0) == 0)
    assert num_of_disconnected_nodes == 0
    return adjacency, features, labels, genres

def save_features_labels_adjacency(use_PCA = True, use_eigenmaps = False, rem_disconnected = True, threshold = 0.66, metric = "correlation",use_features = ['mfcc'], dataset_size = 'small',genres=None,num_classes=None, return_features=False,plot_graph=False, train=True,use_mlp=False,use_cpu=False):
    tracks, features = csv_loader()
    features_part_train, features_part_test, genres_gt_train, genres_gt_test, genres_classes, dict_genres, release_dates = select_features(tracks, features, use_features = use_features, dataset_size = dataset_size, genres =genres, num_classes=num_classes)

    name = form_file_names(use_PCA, use_eigenmaps, rem_disconnected, dataset_size, threshold,use_mlp)

    all_features = np.vstack([features_part_test, features_part_train])
    all_labels = np.concatenate([genres_gt_test, genres_gt_train])
    idx_test = np.array(list(range(0,features_part_test.shape[0])))
    adjacency_big, _, all_labels, _,  = form_adjacency(all_features, all_labels, genres_classes, rem_disconnected, threshold = threshold, metric = metric)

    np.save("dataset_saved_numpy/"+ name + "all_labels.npy", all_labels)
    np.save("dataset_saved_numpy/"+ name + "all_adjacency.npy", adjacency_big)

    for save_bool in [not train, train]:
        if save_bool:
            feature_values = features_part_train
            genres_gt = genres_gt_train
            file_name = name + "train_"
        else:
            feature_values = features_part_test
            genres_gt = genres_gt_test
            file_name = name + "test_"

        if (use_PCA):
            if save_bool:
                feature_values = normalize_feat(feature_values)
                feature_values = generate_PCA_features(feature_values)
            else:
                temp_feat = normalize_feat(all_features)
                temp_feat = generate_PCA_features(temp_feat)
                feature_values = temp_feat[idx_test, :]

        if use_mlp:
            mlp_name = form_file_names(use_PCA, use_eigenmaps, rem_disconnected, dataset_size, threshold,not use_mlp)
            mlp_nn = MLP_NN(hidden_size=100, features=feature_values, labels=genres_gt,num_epoch=10,batch_size=100,num_classes=len(genres_classes), save_path=mlp_name,cuda=use_cpu)
            feature_values = mlp_nn.get_rep(feature_values)
        adjacency, feature_values, genres_gt, genres_classes  = form_adjacency(feature_values, genres_gt, genres_classes, rem_disconnected,  threshold = threshold, metric = metric)
        if (use_eigenmaps):
            feature_values = spectral_embedding(adjacency,n_components=10, eigen_solver=None,random_state=SEED, eigen_tol=0.0,norm_laplacian=True)  

        np.save("dataset_saved_numpy/"+ file_name + "labels.npy", genres_gt)
        np.save("dataset_saved_numpy/"+ file_name + "adjacency.npy", adjacency)
        np.save("dataset_saved_numpy/"+ file_name + "features.npy", feature_values)
        np.save("dataset_saved_numpy/" + file_name + "genres_classes.npy", genres_classes)
        np.save("dataset_saved_numpy/dict_genres.npy", dict_genres)
        np.save("dataset_saved_numpy/" + file_name + "release_dates.npy", release_dates)
        print("Dataset of size {}. Features,labels,and genres saved using prefix: {}".format(adjacency.shape[0], file_name[:len(file_name)-1]))

    if (return_features):
        pygsp_graph = None
        if plot_graph:
            subsampled_adjacency, genres_gt = uniform_random_subsample(adjacency, genres_gt)
            pygsp_graph = pg.graphs.Graph(subsampled_adjacency, lap_type = 'normalized')
            pygsp_graph.set_coordinates('spring') #for visualization
            plot_gt_labels(pygsp_graph, genres_gt, name)
        return feature_values, genres_gt, genres_classes, adjacency, pygsp_graph, release_dates
    return name

def load_transductive_data(name):
    labels = np.load("dataset_saved_numpy/"+ name + "all_labels.npy")
    adjacency = np.load("dataset_saved_numpy/"+ name + "all_adjacency.npy")
    n_data = adjacency.shape[0]
    idx_test = np.array(list(range(0,n_data//10)))
    idx_tr = np.array(list(range(n_data//10, n_data)))
    return labels, adjacency, idx_test, idx_tr

def load_features_labels_adjacency(name, train, plot_graph=False):
    if train:
        file_name = name + "train_"
    else:
        file_name = name + "test_"

    assert os.path.exists("dataset_saved_numpy/" + file_name + "features.npy")
    assert os.path.exists("dataset_saved_numpy/" + file_name + "adjacency.npy")
    assert os.path.exists("dataset_saved_numpy/" + file_name + "labels.npy")
    assert os.path.exists("dataset_saved_numpy/dict_genres.npy")
    assert os.path.exists("dataset_saved_numpy/" + file_name + "genres_classes.npy")
    assert os.path.exists("dataset_saved_numpy/" + file_name + "release_dates.npy")

    features = np.load("dataset_saved_numpy/" + file_name + "features.npy")
    adjacency =  np.load("dataset_saved_numpy/" + file_name + "adjacency.npy")
    labels = np.load("dataset_saved_numpy/" + file_name  + "labels.npy")
    dict_genres = np.load("dataset_saved_numpy/dict_genres.npy")
    genres_classes = np.load("dataset_saved_numpy/" + file_name + "genres_classes.npy")
    release_dates = np.load("dataset_saved_numpy/" + file_name + "release_dates.npy")

    pygsp_graph = None
    if plot_graph:
        subsampled_adjacency, genres_gt = uniform_random_subsample(adjacency, labels)
        pygsp_graph = pg.graphs.Graph(subsampled_adjacency, lap_type = 'normalized')
        pygsp_graph.set_coordinates('spring') #for visualization
        plot_gt_labels(pygsp_graph, genres_gt, name)
    return features, labels, genres_classes, adjacency,  pygsp_graph, release_dates
