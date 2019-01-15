import numpy as np
import pygsp as pg
import pdb
import sys
import argparse

from utils import *
from visualization import *
from models import SVM, Random_Forest, KNN, GCN, MLP
from error import error_func
from graph_analysis import Our_Graph
import graph_stats as gstats
from trainer import Trainer
from evaluate import cross_validation, grid_search_for_param

from sklearn.manifold import spectral_embedding
from dataloader import save_features_labels_adjacency, load_features_labels_adjacency
import transductive as tr
from scipy import sparse

SEED = 0
parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')
parser.add_argument('--only-features', action='store_true',
                    help="Calculate features only (Default:False)")
parser.add_argument('--recalculate-features', action='store_true',
                    help="Calculate features before running classification (Default:False)")
parser.add_argument('--plot-graph', action='store_true',
                    help="Plot Graph (Default:False)")
parser.add_argument('--graph-statistics', type=str, default=None,
                    choices=['basic', 'advanced', 'all'],
                    help="Report Graph Statistics (Default:False)")
parser.add_argument('--with-PCA', action='store_true',
                    help="Apply PCA to features (Default:False)")
parser.add_argument('--use-eigenmaps', action='store_true',
                    help="Use eigenmaps (Default:False)")
parser.add_argument('--genres', default=None, nargs='+', type=str,
                    help="list of genre used(Default: None)")
parser.add_argument('--num-classes', type=int, default=None,
                    help="number of random genres(Default:Use list from --genres)")
parser.add_argument('--threshold', type=float, default=0.95,
                    help="Threshold for cutting weight (Default:0.95)")
parser.add_argument('--distance-metric', type=str, default='correlation',
                    choices=['braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
                    'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 'kulsinski',
                    'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                    'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'])
parser.add_argument('--dataset-size', type=str, default=None,
                    choices=['small', 'medium', 'large'])
parser.add_argument('--transductive-learning', action='store_true',
                    help="Apply Transductive Learning (Default:False)")
parser.add_argument('--inductive-learning', action='store_true',
                    help="Apply Inductive Learning (Default:False)")

def run_demo(args):
    args = parser.parse_args(args)
    np.random.seed(SEED)
    default_name = ""
    pca_name = "normalized_PCA_"
    eigenmaps_name = "eigenmaps_"
    stat_dirname = "graph_stats"

    if args.recalculate_features or args.only_features:
        print("Calculating Features ...")
        if not(args.genres is None) and len(args.genres)>0:
            num_classes = None
        else:
            num_classes = args.num_classes

        output = save_features_labels_adjacency(normalize_features=False, use_PCA=False, rem_outliers=False, threshold=args.threshold, metric=args.distance_metric,
                                       use_features=['mfcc'], dataset_size=args.dataset_size, genres=args.genres, num_classes=num_classes, return_features=args.recalculate_features,plot_graph=args.plot_graph)
        if args.with_PCA:
            output_pca = save_features_labels_adjacency(normalize_features = True, use_PCA = True, rem_outliers= False, threshold =args.threshold, metric=args.distance_metric,
                                           use_features=['mfcc'], dataset_size=args.dataset_size, genres=args.genres, num_classes=num_classes, return_features=args.recalculate_features,plot_graph=args.plot_graph)
        if args.only_features:
            return

        features, gt_labels, gt_labels_onehot, genres, adjacency, full_adjacency, pygsp_graph, release_dates = output
        if args.with_PCA:
            features_pca, gt_labels, gt_labels_onehot, genres, adjacency_pca, full_adjacency_pca, pygsp_graph_pca, release_dates = output_pca
    else:
        print("Loading features, labels, and adjacency")
        features, gt_labels, gt_labels_onehot, genres, adjacency, full_adjacency, pygsp_graph, release_dates = load_features_labels_adjacency(default_name,plot_graph=args.plot_graph)
        if args.with_PCA:
            features_pca, gt_labels, gt_labels_onehot, genres, adjacency_pca, full_adjacency_pca, pygsp_graph_pca, release_dates = load_features_labels_adjacency(pca_name,plot_graph=args.plot_graph)

    print("adjacency", adjacency.shape)
    print("full_adjacency", full_adjacency.shape)

    print("Genres that will be used: {}".format(genres))
    n_data = features.shape[0]

    if args.plot_graph:
        plot_gt_labels(pygsp_graph, gt_labels, default_name)
        if args.with_PCA:
            plot_gt_labels(pygsp_graph_pca, gt_labels, pca_name)

    if args.graph_statistics:
        if not os.path.exists(stat_dirname):
            os.makedirs(stat_dirname)

        if args.graph_statistics == 'all':
            if args.with_PCA:
                gstats.allstats(adjacency_pca, stat_dirname, active_plots=False)
            else:
                gstats.allstats(adjacency, stat_dirname, active_plots=False)
        elif args.graph_statistics == 'advanced':
            if args.with_PCA:
                gstats.advanced(adjacency_pca, stat_dirname, active_plots=args.plot_graph)
            else:
                gstats.advanced(adjacency, stat_dirname, active_plots=args.plot_graph)
        else:  # basic setting
            if args.with_PCA:
                gstats.basic(adjacency_pca)
            else:
                gstats.basic(adjacency)

        if args.with_PCA:
            gstats.growth_analysis(adjacency_pca, release_dates, gt_labels, stat_dirname)
        else:
            gstats.growth_analysis(adjacency, release_dates, gt_labels, stat_dirname)

    if args.transductive_learning:
        print('#### Applying Transductive Learning ####')
        transductive_learning(adjacency=adjacency,labels=gt_labels,genres=genres,n_data=n_data,name=default_name)
        if args.with_PCA:
            print('#### Applying Transductive Learning on PCA Adjacency  ####')
            transductive_learning(adjacency=adjacency_pca,labels=gt_labels,genres=genres,n_data=n_data,name=pca_name)

    if args.inductive_learning:
        print('#### Applying Transductive Learning ####')
        if args.use_eigenmaps:
            features_lap = spectral_embedding(adjacency,n_components=10, eigen_solver=None,
                               random_state=SEED, eigen_tol=0.0,
                               norm_laplacian=True)
            if args.with_PCA:
                features_lap_pca = spectral_embedding(adjacency_pca ,n_components=10, eigen_solver=None,
                                   random_state=SEED, eigen_tol=0.0,
                                   norm_laplacian=True)


        svm_clf = SVM(features, gt_labels, kernel='poly',seed=SEED)
        random_forest_clf = Random_Forest(features, gt_labels, n_estimators=1000, max_depth=2,seed=SEED)
        knn_clf = KNN(features, gt_labels)
        mlp_clf = MLP(features, gt_labels, solver='adam', alpha=1e-5, hidden_layers=(10, 8), lr=2e-4, max_iter=10000)
        if args.with_PCA:
            svm_clf_pca = SVM(features_pca, gt_labels, kernel='poly',seed=SEED)
            random_forest_clf_pca = Random_Forest(features_pca, gt_labels, n_estimators=1000, max_depth=2,seed=SEED)
            knn_clf_pca = KNN(features_pca, gt_labels)


        if args.use_eigenmaps:
            svm_clf_lap = SVM(features_lap, gt_labels, kernel='poly', seed=SEED)
            random_forest_clf_lap = Random_Forest(features_lap, gt_labels, n_estimators=1000, max_depth=2, seed=SEED)
            knn_clf_lap = KNN(features_lap, gt_labels)

            if args.with_PCA:
                svm_clf_lap_pca = SVM(features_lap_pca, gt_labels, kernel='poly', seed=SEED)
                random_forest_clf_lap_pca = Random_Forest(features_lap_pca, gt_labels, n_estimators=1000, max_depth=2, seed=SEED)
                knn_clf_lap_pca = KNN(features_lap_pca, gt_labels)



        #nhid = 100 gives 82.5, nhid=500 gives 83, nhid = 750 gives 83.5 ---> adjacency
        #dropout = 0.1, nhid= 750 gives 86.5, dropout=0.3 and nhid=750 gives 87.25   --> adjacency_pca
        if args.with_PCA:
            print('############## GNN + PCA ##############')
            gnn_clf_pca = GCN(nhid=[750, 100], dropout=0.1, adjacency= adjacency_pca, features=features_pca, labels=gt_labels_onehot, cuda=True, regularization=None, lr=0.01, weight_decay = 5e-4, epochs = 100, batch_size=2000)
            #gnn.train()
            #gnn.classify()
            mean_error_gnn, std_error_gnn = cross_validation(gnn_clf_pca, n_data, K=5,classes=genres, name=pca_name+"gnn_")
        else:
            print('############## GNN ##############')
            gnn_clf = GCN(nhid=[750, 100], dropout=0.1, adjacency= adjacency, features=features, labels=gt_labels_onehot, cuda=True, regularization=None, lr=0.01, weight_decay = 5e-4, epochs = 100, batch_size=2000)
            #gnn.train()
            #gnn.classify()
            mean_error_gnn, std_error_gnn = cross_validation(gnn_clf, n_data, K=5,classes=genres, name=default_name+"gnn_")

        print('* GNN cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_gnn, std_error_gnn))


        print('############## Normal Adjacency ##############')

        mean_error_svm, std_error_svm = cross_validation(svm_clf, n_data,  K=5,classes=genres, name=default_name+"svm_")
        print('* SVM cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_svm, std_error_svm))

        mean_error_rf, std_error_rf = cross_validation(random_forest_clf, n_data,  K=5,classes=genres, name=default_name+"rf_")
        print('* Random Forest cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_rf, std_error_rf))

        mean_error_knn, std_error_knn = cross_validation(knn_clf, n_data,  K=5,classes=genres, name=default_name+"knn_")
        print('* KNN cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_knn, std_error_knn))

        mean_error_mlp, std_error_mlp = cross_validation(mlp_clf, n_data, classes=genres,
                                                         name=default_name + "mlp_")
        print('* MLP cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_mlp, std_error_mlp))
        print('')

        if args.with_PCA:
            print('############## Normalized + PCA ##############')

            mean_error_svm, std_error_svm = cross_validation(features_pca, gt_labels, svm_clf, K=5,classes=genres, name=pca_name+"svm_")
            print('* Normalized, PCA, SVM cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_svm, std_error_svm))

            mean_error_rf, std_error_rf = cross_validation(features_pca, gt_labels, random_forest_clf, K=5,classes=genres, name=pca_name+"rf_")
            print('* Normalized, PCA, Random Forest cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_rf, std_error_rf))

            mean_error_knn, std_error_knn = cross_validation(features_pca, gt_labels, knn_clf, K=5,classes=genres, name=pca_name+"knn_")
            print('* Normalized, PCA, KNN cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_knn, std_error_knn))
            print('')

        if args.use_eigenmaps:
            print('############## Using Eigenmaps ##############')

            mean_error_svm, std_error_svm = cross_validation(features_lap, gt_labels, svm_clf, K=5,classes=genres, name=default_name+eigenmaps_name+"svm_")
            print('* Eigenmaps,SVM cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_svm, std_error_svm))

            mean_error_rf, std_error_rf = cross_validation(features_lap, gt_labels, random_forest_clf, K=5,classes=genres, name=default_name+eigenmaps_name+"rf_")
            print('* Eigenmaps,Random Forest cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_rf, std_error_rf))

            mean_error_knn, std_error_knn = cross_validation(features_lap, gt_labels, knn_clf, K=5,classes=genres, name=default_name+eigenmaps_name+"knn_")
            print('* Eigenmaps,KNN cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_knn, std_error_knn))
            print('')
            if args.with_PCA:
                print('############## Using Eigenmaps + Normalized + PCA ##############')

                mean_error_svm, std_error_svm = cross_validation(features_lap_pca, gt_labels, svm_clf, K=5,classes=genres, name=pca_name+eigenmaps_name+"svm_")
                print('* Normalized, PCA,Eigenmaps, SVM cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_svm, std_error_svm))

                mean_error_rf, std_error_rf = cross_validation(features_lap_pca, gt_labels, random_forest_clf, K=5,classes=genres, name=pca_name+eigenmaps_name+"rf_")
                print('* Normalized, PCA,Eigenmaps, Random Forest cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_rf, std_error_rf))

                mean_error_knn, std_error_knn = cross_validation(features_lap_pca, gt_labels, knn_clf, K=5,classes=genres, name=pca_name+eigenmaps_name+"knn_")
                print('* Normalized, PCA,Eigenmaps, KNN cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_knn, std_error_knn))

def run_grid_search_for_optimal_param():
    pca_name = "normalized_PCA_"
    features_pca, gt_labels, gt_labels_onehot,genres, adjacency_pca, pygsp_graph_pca = load_features_labels_adjacency(pca_name)

    svm_clf = SVM(kernel='poly',seed=SEED)
    random_forest_clf = Random_Forest(n_estimators=1000, max_depth=2,seed=SEED)
    knn_clf = KNN()

    #grid_search_for_param(features_pca, gt_labels, knn_clf, "KNN", K=5, name=pca_name) #ran it and found 42
    grid_search_for_param(features_pca, gt_labels, random_forest_clf, "Random_Forest",classes=genres, K=5, name=pca_name)

def transductive_learning(adjacency,labels,genres,n_data,name):
    adjacency = sparse.csr_matrix(adjacency)

    lgc = tr.LGC(graph=adjacency,y=labels,alpha=0.50,max_iter=30)
    hmn = tr.HMN(graph=adjacency,y=labels,max_iter=30)
    parw = tr.PARW(graph=adjacency,y=labels,lamb=10,max_iter=30)
    mad = tr.MAD(graph=adjacency,y=labels,mu=np.array([1.0,0.5,1.0]),beta=2.0,max_iter=30)
    omni = tr.OMNIProp(graph=adjacency,y=labels,lamb=1.0,max_iter=30)
    camlp = tr.CAMLP(graph=adjacency,y=labels,beta=0.1,H=None,max_iter=30)

    mean_error_lgc, std_error_lgc = cross_validation(lgc, n_data,  K=5,classes=genres, name=name+"lgc_")
    print('* Local and Global Consistency - cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_lgc, std_error_lgc))

    mean_error_hmn, std_error_hmn = cross_validation(hmn, n_data,  K=5,classes=genres, name=name+"hmn_")
    print('* Harmonic Function - cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_hmn, std_error_hmn))

    mean_error_parw, std_error_parw = cross_validation(parw, n_data,  K=5,classes=genres, name=name+"parw_")
    print('* Partially Absorbing Random Walk - cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_parw, std_error_parw))

    mean_error_mad, std_error_mad = cross_validation(mad, n_data,  K=5,classes=genres, name=name+"mad_")
    print('* Modified Adsorption - cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_mad, std_error_mad))

    mean_error_omni, std_error_omni = cross_validation(omni, n_data,  K=5,classes=genres, name=name+"omni_")
    print('* OMNI-Prop - cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_omni, std_error_omni))

    mean_error_camlp, std_error_camlp = cross_validation(camlp, n_data,  K=5,classes=genres, name=name+"camlp_")
    print('* Confidence-Aware Modulated Label Propagation - cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_camlp, std_error_camlp))


if __name__ == "__main__":
    run_demo(sys.argv[1:])