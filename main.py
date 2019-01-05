import numpy as np
import pygsp as pg
import pdb

from dataloader import *
from utils import *
from visualization import *
from models import *
from error import error_func
from graph_analysis import Our_Graph
from sklearn.decomposition import PCA


if __name__ == "__main__":
    adjacency, adjacency_pg = load_adjacency_matrix_from_npy()
    features = load_features_selected_from_numpy()
    gt_labels = load_labels_from_npy()
    graph = Our_Graph(adjacency)

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
        features_tr = features[mask_pos]
        features_test = features[test_ind]
        gt_labels_tr = gt_labels[mask_pos].squeeze()
        gt_labels_test = gt_labels[test_ind].squeeze()

        #####################
        #Spectral Clustering#
        #####################
        graph.spectral_clustering(test_ind, gt_labels_test)

        ##############################################################################################################
        #Transductive learning by minimizing a (semi-) p-norm of the graph gradient applied to the signal of interest#
        ##############################################################################################################
        n_trials = 10
        p=1

        graph.transductive_learning(w, thresholds, n_trials, gt_labels, gt_labels_test, p)
        p=2
        graph.transductive_learning(w, thresholds, n_trials, gt_labels, gt_labels_test, p)

        #####
        #SVM#
        #####
        svm_clf = SVM()
        svm_clf.train(features_tr, gt_labels_tr)
        predicted_classes_svm = svm_clf.classify(features_test)
        print('SVM Percentage Error: {:.2f}'.format(error_func(gt_labels_test, predicted_classes_svm)))

        ###########
        #SVM + PCA#
        ###########
        svm_pca_clf = SVM()
        pca = PCA(n_components=2, svd_solver='arpack')
        features_pca_tr = pca.fit_transform(features_tr)
        features_pca_test = pca.transform(features_test)
        svm_clf.train(features_pca_tr, gt_labels_tr)
        predicted_classes_svm_pca = svm_clf.classify(features_pca_test)
        print('SVM + PCA Percentage Error: {:.2f}'.format(error_func(gt_labels_test, predicted_classes_svm)))

    plot_gt_labels(adjacency_pg, gt_labels)
