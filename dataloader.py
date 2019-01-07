import pandas as pd
import numpy as np
import ast
from scipy.spatial.distance import pdist, squareform
from utils import *
import pdb
import pygsp as pg


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

def select_features(tracks, features, use_features = ['mfcc'], dataset_size = 'small', genres = ['Hip-Hop', 'Rock']):
    small = tracks[tracks['set', 'subset'] == dataset_size]

    # Filters out only the tracks that are Hip-Hop and Rock from small subset --> 2000 tracks
    if genres:
        temp = small['track', 'genre_top'] == genres[0]
        for ind in range(1, len(genres)):
            temp = temp | (small['track', 'genre_top'] == genres[ind])
        subset = small[temp]

    # Takes a subset of features based on the tracks found in the variable subset
    small_features = features[features.index.isin(subset.index)] 
    #features_part_mfcc = small_features.loc[:,('mfcc', ('median', 'mean'), slice('01','12'))] 
    #features_part_chroma = small_features.loc[:,('chroma_cens', ('median', 'mean'), slice('01','05'))] # Take chroma column as features
    #features_part = features_part_mfcc.join(features_part_chroma)
    features_part = small_features.loc[:,use_features].values

    #save labels    
    genres_gt = np.zeros([features_part.shape[0],1])
    for ind in range(0, len(genres)):
        temp = (subset['track', 'genre_top'] == genres[ind]).to_frame().values
        genres_gt[temp] = ind
    return features_part, genres_gt

def form_adjacency(features, threshold = 0.66, metric ='correlation'):
    distances = pdist(features, metric=metric) # Use cosine equation to calculate distance 
    kernel_width = distances.mean()
    weights = np.exp(-distances**2 / kernel_width**2) # Use Gaussian function to calculate weights

    weights[weights < threshold] = 0
    adjacency = squareform(weights)
    num_of_disconnected_nodes = np.sum(np.sum(adjacency, axis=0) == 0)
    assert num_of_disconnected_nodes == 0
    return adjacency

def save_features_labels_adjacency(normalize_features = True, use_PCA = True, rem_outliers = True):
    tracks, features = csv_loader()
    feature_values, genres_gt = select_features(tracks, features, use_features = ['mfcc'], dataset_size = 'small', genres = ['Hip-Hop', 'Rock'])
    np.save("labels.npy", genres_gt)

    name = ""
    if (normalize_features):
        feature_values = normalize_feat(feature_values)
        name += "normalized_"
    if use_PCA:    
        feature_values = generate_PCA_features(feature_values)
        name += "PCA_"
    if rem_outliers:
        feature_values = remove_outliers(feature_values)
        name += "nooutlier_"
    
    adjacency = form_adjacency(feature_values, threshold = 0.66, metric = "correlation")
    np.save(name + "adjacency.npy", adjacency)
    np.save(name + "features.npy", feature_values)
    return name

def load_features_labels_adjacency(name):
    features = np.load(name + "features.npy")
    adjacency =  np.load(name + "adjacency.npy")
    labels = np.load("labels.npy")

    adjacency_pg = pg.graphs.Graph(adjacency, lap_type = 'normalized')
    adjacency_pg.set_coordinates('spring') #for visualization
    return features, labels, adjacency, adjacency_pg

if __name__ == "__main__":
    name1 = save_features_labels_adjacency(normalize_features = False, use_PCA = False, rem_outliers= False)
    name2 = save_features_labels_adjacency(normalize_features = True, use_PCA = True, rem_outliers= False)
    print(name1)
