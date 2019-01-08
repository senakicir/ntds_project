import numpy as np
import pygsp as pg
import pdb

from dataloader import *
from utils import *
from visualization import *
from models import *
from error import error_func
from graph_analysis import Our_Graph
from trainer import Trainer
from evaluate import cross_validation

if __name__ == "__main__":
    default_name = ""
    pca_name = "normalized_PCA_"

    #features, gt_labels, adjacency, adjacency_pg = load_features_labels_adjacency(default_name)
    features_pca, gt_labels, adjacency_pca, adjacency_pg_pca = load_features_labels_adjacency(pca_name)
    #plot_gt_labels(adjacency_pg, gt_labels, default_name)
    plot_gt_labels(adjacency_pg_pca, gt_labels, pca_name)

    #graph = Our_Graph(adjacency_pca)
    #features_lap = graph.get_laplacian_eigenmaps()
    #adjacency_pg.get_laplacian_eigenmaps

    svm_clf = SVM()
    mean_error, std_error = cross_validation(features_pca, gt_labels, svm_clf, K=5, name=pca_name)
    print('Cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error, std_error))


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

