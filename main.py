import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import pygsp as pg
import pdb
import sys
import argparse

from utils import *
from visualization import *
from models import SVM, Random_Forest, KNN, GCN, MLP, GCN_KHop
from error import error_func
from graph_analysis import Our_Graph
import graph_stats as gstats
from trainer import Trainer
from evaluate import cross_validation, simple_test, evaluate_transductive, train_gcn

from dataloader import save_features_labels_adjacency, load_features_labels_adjacency
import transductive as tr
from scipy import sparse
import time as time
from mlp import MLP as MLP_NN

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
parser.add_argument('--PCA-dim', type=int, default=10,
                    help="Choose the number of dimensions to reduce to with the PCA")
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
parser.add_argument('--use-cpu', action='store_false',
                    help="Use CPU when training the GCN (Default:False)")
parser.add_argument('--gcn', action='store_true',
                    help="Evaluate GCN (Default:False)")
parser.add_argument('--gcn_khop', action='store_true',
                    help="Evaluate GCN KHOP (Default:False)")
parser.add_argument('--mlp-nn', action='store_true',
                    help="Evaluate MLPNN (Default:False)")
parser.add_argument('--use-mlp-features', action='store_true',
                    help="use mlp features (Default:False)")
parser.add_argument('--additional-models', action='store_true',
                    help="Evaluate with SVM, RBF, KNN, KMeans, MLP (Default:False)")
parser.add_argument('--remove-disconnected', action='store_true',
                    help="Remove outliers (Default:False)")
parser.add_argument('--train', action='store_true',
                    help="Trains all models and evaluates them using cross validation  (Default:False)")
parser.add_argument('--prefix', type=str, default="None",
                    help="Add prefix to fileName")

def load_parameters_and_data(args):
    stat_dirname = "graph_stats"
    ## Form prefixes for saved files depending on the arguments passed
    names = form_file_names(args.with_PCA, args.PCA_dim, args.use_eigenmaps, args.remove_disconnected, args.dataset_size, args.threshold,args.use_mlp_features,args.prefix)

    if args.recalculate_features or args.only_features:
        print("Calculating Features ...")
        if not(args.genres is None) and len(args.genres)>0:
            num_classes = None
        else:
            num_classes = args.num_classes

        ## Read the features and labels, divide into training and testing, for adjacency matrix and save them
        output = save_features_labels_adjacency(use_PCA=args.with_PCA, PCA_dim = args.PCA_dim, use_eigenmaps=args.use_eigenmaps, rem_disconnected= args.remove_disconnected, threshold =args.threshold, metric=args.distance_metric,
                                           use_features=['mfcc'], dataset_size=args.dataset_size, genres=args.genres, num_classes=num_classes, return_features=args.recalculate_features,plot_graph=args.plot_graph,train=args.train, use_mlp=args.use_mlp_features,use_cpu=args.use_cpu,prefix=args.prefix)

        if args.only_features:
            return
        features, gt_labels, genres, adjacency,indx_train,indx_test, pygsp_graph, release_dates = output
    else:
        print("Loading features, labels, and adjacency ...")
        ## Load previously saved features, labels, and adjacency matrices
        features, gt_labels, genres, adjacency,indx_train,indx_test, pygsp_graph, release_dates = load_features_labels_adjacency(names, args.train, plot_graph=args.plot_graph)

    print("The dataset size is: {}. Genres that will be used: {}".format(features.shape[0], genres))

    return args, names, stat_dirname, features, gt_labels, genres, adjacency,indx_train,indx_test, pygsp_graph, release_dates

def train_everything(args):
    ## Get features, labels, training and testing set, adjacency
    args, file_names, stat_dirname, features, gt_labels, genres, adjacency,indx_train,indx_test, pygsp_graph, release_dates = load_parameters_and_data(args)

    if args.inductive_learning:
        print('#### Applying Inductive Learning ####')

        if args.additional_models:
            ## Initialize model with correct parameters
            svm_clf = SVM(features, gt_labels, kernel='linear', seed=SEED, save_path=file_names)
            random_forest_clf = Random_Forest(features, gt_labels, n_estimators=100, max_depth=20,seed=SEED, save_path=file_names)
            knn_clf = KNN(features, gt_labels, save_path=file_names)

            start = time.time()
            mean_error_svm, std_error_svm = cross_validation(svm_clf, indx_train, K=5, classes=genres, name=file_names+"svm_")
            print('* SVM cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_svm, std_error_svm))
            print("SVM time", time.time()-start)

            start = time.time()
            mean_error_rf, std_error_rf = cross_validation(random_forest_clf, indx_train, K=5,classes=genres, name=file_names+"rf_")
            print('* Random Forest cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_rf, std_error_rf))
            print("RF time", time.time()-start)

            start = time.time()
            mean_error_knn, std_error_knn = cross_validation(knn_clf, indx_train, K=5,classes=genres, name=file_names+"knn_")
            print('* KNN cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_knn, std_error_knn))
            print("KNN time", time.time()-start)

        if args.gcn:
            print("Training GCN")
            start = time.time()
            ## Initialize GCN with correct parameters
            gnn_clf = GCN(nhid=[1200, 100], dropout=0.1, adjacency= adjacency, features=features, labels=gt_labels, n_class=len(genres), cuda=args.use_cpu, regularization=None, lr=0.01, weight_decay = 5e-4, epochs = 300, batch_size=10000, save_path=file_names)
            train_gcn(gnn_clf, indx_train, name=file_names+"gnn_")
            print("GCN time", time.time()-start)

        if args.gcn_khop:
            print("Training GCN K-Hop")
            start = time.time()
            ## Initialize GCN K-Hop with correct parameters
            gnn_clf = GCN_KHop(nhid=[1200, 100], dropout=0.1, adjacency=adjacency, features=features, labels=gt_labels,
                               n_class=len(genres), khop=2, cuda=args.use_cpu, regularization=None, lr=0.01,
                               weight_decay=5e-4, epochs=300, batch_size=10000, save_path=file_names)
            train_gcn(gnn_clf, indx_train, name=file_names + "gnn_khop_")
            print("GCN K-Hop time", time.time()-start)

        if args.mlp_nn:
            start = time.time()
            ## Initialize MLP with correct parameters
            mlp_nn = MLP_NN(hidden_size=100, features=features, labels=gt_labels,num_epoch=100,batch_size=100,num_classes=len(genres), save_path=file_names,cuda=args.use_cpu)
            mean_error_mlpNN, std_error_mlpNN = cross_validation(mlp_nn, indx_train, K=5,classes=genres, name=file_names+"mlpNN_")
            print('* MLP NN cross validation error mean: {:.2f}, std: {:.2f}'.format(mean_error_mlpNN, std_error_mlpNN))
            print("MLP time", time.time()-start)

def test_everything(args):
    ## Get features, labels, training and testing set, adjacency
    args, file_names, stat_dirname, features, gt_labels, genres, adjacency,indx_train,indx_test, pygsp_graph, release_dates = load_parameters_and_data(args)

    if args.graph_statistics:
        if not os.path.exists(stat_dirname):
            os.makedirs(stat_dirname)

        if args.graph_statistics == 'all':
            ## Prints out all statistics about graph
            gstats.allstats(adjacency, stat_dirname, active_plots=False)
        elif args.graph_statistics == 'advanced':
            ## Prints out all advanced statistics
            gstats.advanced(adjacency, stat_dirname, active_plots=args.plot_graph)
        else:  # basic setting
            ## Prints out basic statistics
            gstats.basic(adjacency)
        gstats.growth_analysis(adjacency, release_dates, gt_labels, stat_dirname)

    if args.inductive_learning:
        print('#### Testing Inductive Learning ####')
        if args.additional_models:
            ## Initialize models with correct parameters
            svm_clf = SVM(features, gt_labels, kernel='linear',seed=SEED, save_path=file_names)
            random_forest_clf = Random_Forest(features, gt_labels, n_estimators=100, max_depth=20,seed=SEED, save_path=file_names)
            knn_clf = KNN(features, gt_labels, save_path=file_names)

            error_svm = simple_test(svm_clf, indx_test, classes=genres, name=file_names+"svm_")
            print('* SVM simple test error: {:.2f}'.format(error_svm))

            error_rf = simple_test(random_forest_clf, indx_test, classes=genres, name=file_names+"rf_")
            print('* Random Forest simple test error: {:.2f}'.format(error_rf))

            error_knn = simple_test(knn_clf, indx_test, classes=genres, name=file_names+"knn_")
            print('* KNN simple test error: {:.2f}'.format(error_knn))

        if args.gcn:
            ## Initialize GCN with correct parameters
            gnn_clf = GCN(nhid=[1200, 100], dropout=0.1, adjacency= adjacency, features=features, labels=gt_labels, n_class=len(genres), cuda=args.use_cpu, regularization=None, lr=0.01, weight_decay = 5e-4, epochs = 300, batch_size=10000, save_path=file_names)
            error_gnn = simple_test(gnn_clf, indx_test, classes=genres, name=file_names+"gnn_")
            print('* GCN simple test error: {:.2f}'.format(error_gnn))
        if args.gcn_khop:
            ## Initialize GCN K-Hop with correct parameters
            gnn_clf = GCN_KHop(nhid=[1200, 100], dropout=0.1, adjacency= adjacency, features=features, labels=gt_labels, n_class=len(genres), khop=2, cuda=args.use_cpu, regularization=None, lr=0.01, weight_decay = 5e-4, epochs = 300, batch_size=10000, save_path=file_names)
            error_gnn = simple_test(gnn_clf, indx_test, classes=genres, name=file_names+"gnn_khop_")
            print('* GCN KHop simple test error: {:.2f}'.format(error_gnn))
        if args.mlp_nn:
            ## Initialize MLP with correct parameters
            mlp_nn = MLP_NN(hidden_size=100, features=features, labels=gt_labels,num_epoch=10,batch_size=100,num_classes=len(genres), save_path=file_names,cuda=args.use_cpu)
            error_mlpNN = simple_test(mlp_nn, indx_test, classes=genres, name=file_names+"mlpNN_")
            print('* MLP NN simple test error: {:.2f}'.format(error_mlpNN))

def transductive_learning(args):
    print('#### Applying Transductive Learning ####')
    ## Get features, labels, training and testing set, adjacency
    args, file_names, stat_dirname, features, gt_labels, genres, adjacency,indx_train,indx_test, pygsp_graph, release_dates = load_parameters_and_data(args)

    adjacency = sparse.csr_matrix(adjacency)

    ## Initialize all models
    hmn = tr.HMN(graph=adjacency,y=gt_labels,max_iter=30)
    parw = tr.PARW(graph=adjacency,y=gt_labels,lamb=10,max_iter=30)
    omni = tr.OMNIProp(graph=adjacency,y=gt_labels,lamb=1.0,max_iter=30)

    start = time.time()
    ## Evaluate HMN
    mean_error_hmn = evaluate_transductive(hmn, indx_train, indx_test,  classes=genres, name=file_names+"hmn_")
    print('* Harmonic Function -  error mean: {:.2f}'.format(mean_error_hmn))
    print("Harmonic time", time.time()-start)

    start = time.time()
    ## Evaluate PARW
    mean_error_parw = evaluate_transductive(parw, indx_train, indx_test,  classes=genres, name=file_names+"parw_")
    print('* Partially Absorbing Random Walk -  error mean: {:.2f}'.format(mean_error_parw))
    print("Partially Absorbing Random Walk time", time.time()-start)

    start = time.time()
    ## Evaluate OMNI-PROP
    mean_error_omni = evaluate_transductive(omni, indx_train, indx_test,  classes=genres, name=file_names+"omni_")
    print('* OMNI-Prop -  error mean: {:.2f}'.format(mean_error_omni))
    print("OMNI-Prop time", time.time()-start)

if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    if args.only_features:
        load_parameters_and_data(args)
    else:
        if args.transductive_learning:
            transductive_learning(args)
        else:
            if args.train:
                train_everything(args)
            else:
                test_everything(args)
