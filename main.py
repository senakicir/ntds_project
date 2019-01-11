import numpy as np
import pygsp as pg
import pdb
import sys
import argparse

from utils import *
from visualization import *
from models import SVM, Random_Forest, KNN, GCN
from error import error_func
from graph_analysis import Our_Graph
from trainer import Trainer
from evaluate import cross_validation, grid_search_for_param

from sklearn.manifold import spectral_embedding
from dataloader import save_features_labels_adjacency, load_features_labels_adjacency

SEED = 0
parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
parser.add_argument('--only-features', action='store_true',
                    help="Calculate features only (Default:False)")
parser.add_argument('--recalculate-features', action='store_true',
                    help="Calculate features before running classification (Default:False)")
parser.add_argument('--plot-graph', action='store_true',
                    help="Plot Graph (Default:False)")
parser.add_argument('--genres', default=None, nargs='+', type=str,
                    help="list of genre used(Default: None)")
parser.add_argument('--num-classes', type=int, default=None,
                    help="number of random genres(Default:Use list from --genres)")
parser.add_argument('--threshold', type=float, default=0.66,
                    help="Threshold for cutting weight (Default:0.66)")
parser.add_argument('--distance-metric', type=str, default='correlation',
                    choices=['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
                    'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
                    'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                    'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'])
parser.add_argument('--dataset-size', type=str, default='small',
                    choices=['small', 'medium', 'large'])

def run_demo(args):
    args = parser.parse_args(args)
    np.random.seed(SEED)
    default_name = ""
    pca_name = "normalized_PCA_"
    eigenmaps_name = "eigenmaps_"
    if args.recalculate_features or args.only_features:
        print("Calculating Features ...")
        if not(args.genres is None) and len(args.genres)>0:
            num_classes = None
        else:
            num_classes = args.num_classes

        output = save_features_labels_adjacency(normalize_features=False, use_PCA=False, rem_outliers=False, threshold=args.threshold, metric=args.distance_metric,
                                       use_features=['mfcc'], dataset_size=args.dataset_size, genres=args.genres, num_classes=num_classes, return_features=args.recalculate_features)

        output_pca = save_features_labels_adjacency(normalize_features = True, use_PCA = True, rem_outliers= False, threshold =args.threshold, metric=args.distance_metric,
                                       use_features=['mfcc'], dataset_size=args.dataset_size, genres=args.genres, num_classes=num_classes, return_features=args.recalculate_features)
        if args.only_features:
            return

        features, gt_labels, gt_labels_onehot, genres, adjacency, pygsp_graph = output
        features_pca, gt_labels, gt_labels_onehot, genres, adjacency_pca, pygsp_graph_pca = output_pca
    else:
        print("Loading features, labels, and adjacency")
        features, gt_labels, gt_labels_onehot, genres, adjacency, pygsp_graph = load_features_labels_adjacency(default_name,plot_graph=args.plot_graph)
        features_pca, gt_labels, gt_labels_onehot, genres, adjacency_pca, pygsp_graph_pca = load_features_labels_adjacency(pca_name,plot_graph=args.plot_graph)



    print("Genres that will be used: {}".format(genres))

    if args.plot_graph:
        plot_gt_labels(pygsp_graph, gt_labels, default_name)
        plot_gt_labels(pygsp_graph_pca, gt_labels, pca_name)


    features_lap = spectral_embedding(adjacency,n_components=10, eigen_solver=None,
                       random_state=SEED, eigen_tol=0.0,
                       norm_laplacian=True)

    features_lap_pca = spectral_embedding(adjacency_pca ,n_components=10, eigen_solver=None,
                       random_state=SEED, eigen_tol=0.0,
                       norm_laplacian=True)

    svm_clf = SVM(kernel='poly',seed=SEED)
    random_forest_clf = Random_Forest(n_estimators=1000, max_depth=2,seed=SEED)
    knn_clf = KNN()
    #nhid = 100 gives 82.5, nhid=500 gives 83, nhid = 750 gives 83.5 ---> adjacency
    #dropout = 0.1, nhid= 750 gives 86.5, dropout=0.3 and nhid=750 gives 87.25   --> adjacency_pca
    gnn = GCN(nhid=[750, 100], dropout=0.1, adjacency= adjacency_pca, features=features_pca, labels=gt_labels_onehot, cuda=True, regularization=None, lr=0.01, weight_decay = 5e-4, epochs = 500)
    print('##############GNN##############')
    gnn.train()
    gnn.classify()
    print('############## Normal Adjacency ##############')
    mean_error_svm, std_error_svm = cross_validation(features, gt_labels, svm_clf, K=5,classes=genres, name=default_name+"svm_")
    print('SVM cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_svm, std_error_svm))

    mean_error_rf, std_error_rf = cross_validation(features, gt_labels, random_forest_clf, K=5,classes=genres, name=default_name+"rf_")
    print('Random Forest cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_rf, std_error_rf))

    mean_error_knn, std_error_knn = cross_validation(features, gt_labels, knn_clf, K=5,classes=genres, name=default_name+"knn_")
    print('KNN cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_knn, std_error_knn))
    print('')

    print('############## Normalized + PCA ##############')

    mean_error_svm, std_error_svm = cross_validation(features_pca, gt_labels, svm_clf, K=5,classes=genres, name=pca_name+"svm_")
    print('Normalized, PCA, SVM cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_svm, std_error_svm))

    mean_error_rf, std_error_rf = cross_validation(features_pca, gt_labels, random_forest_clf, K=5,classes=genres, name=pca_name+"rf_")
    print('Normalized, PCA, Random Forest cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_rf, std_error_rf))

    mean_error_knn, std_error_knn = cross_validation(features_pca, gt_labels, knn_clf, K=5,classes=genres, name=pca_name+"knn_")
    print('Normalized, PCA, KNN cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_knn, std_error_knn))
    print('')

    print('############## Using Eigenmaps ##############')

    mean_error_svm, std_error_svm = cross_validation(features_lap, gt_labels, svm_clf, K=5,classes=genres, name=default_name+eigenmaps_name+"svm_")
    print('Eigenmaps,SVM cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_svm, std_error_svm))

    mean_error_rf, std_error_rf = cross_validation(features_lap, gt_labels, random_forest_clf, K=5,classes=genres, name=default_name+eigenmaps_name+"rf_")
    print('Eigenmaps,Random Forest cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_rf, std_error_rf))

    mean_error_knn, std_error_knn = cross_validation(features_lap, gt_labels, knn_clf, K=5,classes=genres, name=default_name+eigenmaps_name+"knn_")
    print('Eigenmaps,KNN cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_knn, std_error_knn))
    print('')

    print('############## Using Eigenmaps + Normalized + PCA ##############')

    mean_error_svm, std_error_svm = cross_validation(features_lap_pca, gt_labels, svm_clf, K=5,classes=genres, name=pca_name+eigenmaps_name+"svm_")
    print('Normalized, PCA,Eigenmaps, SVM cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_svm, std_error_svm))

    mean_error_rf, std_error_rf = cross_validation(features_lap_pca, gt_labels, random_forest_clf, K=5,classes=genres, name=pca_name+eigenmaps_name+"rf_")
    print('Normalized, PCA,Eigenmaps, Random Forest cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_rf, std_error_rf))

    mean_error_knn, std_error_knn = cross_validation(features_lap_pca, gt_labels, knn_clf, K=5,classes=genres, name=pca_name+eigenmaps_name+"knn_")
    print('Normalized, PCA,Eigenmaps, KNN cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_knn, std_error_knn))

def run_grid_search_for_optimal_param():
    pca_name = "normalized_PCA_"
    features_pca, gt_labels, gt_labels_onehot,genres, adjacency_pca, pygsp_graph_pca = load_features_labels_adjacency(pca_name)

    svm_clf = SVM(kernel='poly',seed=SEED)
    random_forest_clf = Random_Forest(n_estimators=1000, max_depth=2,seed=SEED)
    knn_clf = KNN()

    #grid_search_for_param(features_pca, gt_labels, knn_clf, "KNN", K=5, name=pca_name) #ran it and found 42
    grid_search_for_param(features_pca, gt_labels, random_forest_clf, "Random_Forest",classes=genres, K=5, name=pca_name)

if __name__ == "__main__":
    run_demo(sys.argv[1:])


#THIS FUNCTION WILL BE USELESS SOON BUT DO NOT DELETE
def lol():
    default_name = ""
    pca_name = "normalized_PCA_"

    features, gt_labels, adjacency, adjacency_pg = load_features_labels_adjacency(default_name)
    features_pca, gt_labels, adjacency_pca, adjacency_pg_pca = load_features_labels_adjacency(pca_name)
    plot_gt_labels(adjacency_pg, gt_labels, default_name)
    plot_gt_labels(adjacency_pg_pca, gt_labels, pca_name)

    graph = Our_Graph(adjacency_pca)
    features_lap = graph.get_laplacian_eigenmaps()

    ########Split the data into training and test data########
    mask_ratio = [0.3, 0.6]
    thresholds = [0]

    for mr in mask_ratio:
        w = np.random.binomial(n=1, p=mr, size=graph.n_nodes)
        m = sum(w)  # Number of measurements
        print('Sampled {} out of {} nodes'.format(m, graph.n_nodes))
        mask_pos = np.where(w == 1)
        all_ind = np.arange(w.shape[0])
        test_ind = list(set(list(all_ind)).difference(mask_pos[0]))

        features_tr = features_lap[mask_pos]
        features_test = features_lap[test_ind]
        features_pca_tr = features_pca[mask_pos]
        features_pca_test = features_pca[test_ind]

        gt_labels_tr = gt_labels[mask_pos].squeeze()
        gt_labels_test = gt_labels[test_ind].squeeze()

        #####################
        #Spectral Clustering#
        #####################
        #graph.spectral_clustering(test_ind, gt_labels_test)

        ##############################################################################################################
        #Transductive learning by minimizing a (semi-) p-norm of the graph gradient applied to the signal of interest#
        ##############################################################################################################
        #n_trials = 10
        #p=1
        #graph.transductive_learning(w, thresholds, n_trials, gt_labels, test_ind, p)
        #p=2
        #graph.transductive_learning(w, thresholds, n_trials, gt_labels, test_ind, p)

        #####
        #SVM#
        #####
        svm_clf = SVM()
        svm_clf.train(features_tr, gt_labels_tr)
        predicted_classes_svm = svm_clf.classify(features_test)
        plot_confusion_matrix(predicted_classes_svm, gt_labels_test, ['Hip-Hop', 'Rock'], default_name)
        print('SVM Percentage Error: {:.2f}'.format(error_func(gt_labels_test, predicted_classes_svm)))

        ###########m
        #SVM + PCA#
        ###########
        svm_pca_clf = SVM()
        svm_pca_clf.train(features_pca_tr, gt_labels_tr)
        predicted_classes_svm_pca = svm_pca_clf.classify(features_pca_test)
        plot_confusion_matrix(predicted_classes_svm_pca, gt_labels_test, ['Hip-Hop', 'Rock'], pca_name)
        print('SVM + PCA Percentage Error: {:.2f}'.format(error_func(gt_labels_test, predicted_classes_svm_pca)))
