import numpy as np
import pdb
import math
import scipy.sparse as sp
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn import BatchNorm1d, ReLU, Dropout
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from trainer import Trainer
from error import accuracy_prob, error_func
from collections import OrderedDict
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphNeuralNet(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GraphNeuralNet, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid[0])
        self.bn1 = BatchNorm1d(nhid[0])
        self.gc2 = GraphConvolution(nhid[0], nhid[1])
        self.bn2 = BatchNorm1d(nhid[1])
        self.gc3 = GraphConvolution(nhid[1], nclass)
        self.dropout = dropout

        self.model = OrderedDict([
            ("layer_one", self.gc1),
            ("layer_two", self.gc2),
            ("layer_three", self.gc3)])

    def forward(self, x, adj):
        x = F.relu(self.bn1(self.gc1(x, adj)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.bn2(self.gc2(x, adj)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)

class GCN():
    def __init__(self, nhid, dropout, adjacency, features, labels, cuda=True, regularization=None, lr=0.01, weight_decay=5e-4, epochs=100, batch_size=100, save_path=""):
        self.adjacency = adjacency
        self.features = features
        self.nfeat = features.shape[-1]
        self.nhid = nhid
        self.nclass = labels.shape[-1]
        self.epochs = epochs
        self.gcn = GraphNeuralNet(self.nfeat, self.nhid, self.nclass, dropout)

        self.features = torch.FloatTensor(np.array(self.features))
        self.adjacency = (self.adjacency + np.eye(self.adjacency.shape[0]))
        self.D = np.diag(self.adjacency.sum(axis=1))
        self.D_norm = (np.power(np.linalg.inv(self.D), 0.5))
        self.adjacency = self.D_norm @ (self.adjacency) @ self.D_norm
        #self.adjacency = sp.coo_matrix(self.adjacency)
        #self.adjacency = sparse_mx_to_torch_sparse_tensor(self.adjacency)
        self.labels = torch.LongTensor(labels)
        self.model_path = 'models/best_model_' + save_path + 'batch_size_' + str(batch_size) +'_gcn.sav'

        #Create trainer
        self.trainer = Trainer(self.gcn, self.adjacency, self.features, self.labels, cuda, regularization, lr, weight_decay, batch_size, self.model_path)

    def train(self, idx_train):
        train_size = idx_train.shape[0]
        valid_size = train_size // 5
        idx_val = np.random.permutation(np.arange(train_size))[0:valid_size]
        self.idx_train = torch.LongTensor(idx_train)
        self.idx_val = torch.LongTensor(idx_val)
        for epoch in range(self.epochs):
            self.trainer.train(epoch, self.idx_train, self.idx_val)

    def load_pretrained(self):
        #Load a pretrained model to test
        self.gcn.load_state_dict(torch.load(self.model_path))

    def classify(self, idx_test):
        self.labels_test = self.labels[idx_test]
        self.prediction = self.trainer.test(idx_test)

    def accuracy(self, classes):
        c_m = confusion_matrix(self.labels_test, np.argmax(self.prediction.cpu().detach().numpy(),axis=1))
        acc_test = error_func(np.argmax(self.prediction.cpu().detach().numpy(),axis=1), self.labels_test.numpy())
        for i in range(len(classes)):
            labels_count = np.sum(self.labels_test.numpy() == i)
            c_m[i,:] = (c_m[i,:] /labels_count)*100
        return c_m, acc_test

    def reset(self):
        dict_model = self.gcn.model
        for k, v in dict_model.items():
            if "layer" in k:
                dict_model[k].reset_parameters()

class SVM():
    def __init__(self, features, labels, kernel, poly_degree=3, seed=0, save_path=""):
        self.features = features
        self.labels = labels
        self.kernel = kernel
        self.poly_degree = poly_degree
        self.seed = seed
        if kernel == 'linear':
            self.clf = svm.LinearSVC(multi_class='ovr',random_state=seed) # Can use 'crammer_singer’ but more expensive while not that much better accuracy(only more stable)
        else:
            self.clf = svm.SVC(gamma='auto', kernel=kernel, degree=poly_degree, decision_function_shape='ovr',random_state=seed)
        self.model_path = 'models/best_model_' + save_path + 'svm.sav'
        #Load a pretrained model to test

    def train(self, idx_train):
        features_tr = self.features[idx_train]
        labels_tr = self.labels[idx_train]
        self.clf.fit(features_tr, labels_tr)

    def load_pretrained(self):
        #Load a pretrained model to test
        self.clf = joblib.load(self.model_path)

    def classify(self, idx_test):
        self.features_test = self.features[idx_test]
        self.labels_test = self.labels[idx_test]
        self.prediction = self.clf.predict(self.features_test)
        joblib.dump(self.clf, self.model_path)

    def accuracy(self, classes):
        c_m = confusion_matrix(self.labels_test, self.prediction)
        for i in range(len(classes)):
            labels_count = np.sum(self.labels_test == i)
            c_m[i,:] = (c_m[i,:] /labels_count)*100
        return c_m, error_func(self.labels_test, self.prediction)

    def reset(self):
        self.clf = svm.SVC(gamma='auto', kernel=self.kernel, degree=self.poly_degree, decision_function_shape='ovr',
                           random_state=self.seed)

class KNN():
    def __init__(self, features, labels, n_neighbors=42, save_path=""):
        self.features = features
        self.labels = labels
        self.n_neighbors = n_neighbors
        self.clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model_path = 'models/best_model_' + save_path + 'knn.sav'
        #Load a pretrained model to test

    def load_pretrained(self):
        #Load a pretrained model to test
        self.clf = joblib.load(self.model_path)

    def train(self, idx_train):
        features_tr = self.features[idx_train]
        labels_tr = self.labels[idx_train]
        self.clf.fit(features_tr, labels_tr)
        joblib.dump(self.clf, self.model_path)

    def classify(self, idx_test):
        self.features_test = self.features[idx_test]
        self.labels_test = self.labels[idx_test]
        self.prediction = self.clf.predict(self.features_test)

    def accuracy(self, classes):
        c_m = confusion_matrix(self.labels_test, self.prediction)
        for i in range(len(classes)):
            labels_count = np.sum(self.labels_test == i)
            c_m[i,:] = (c_m[i,:] /labels_count)*100
        return c_m, error_func(self.labels_test, self.prediction)

    def reset(self):
        self.clf = KNeighborsClassifier(n_neighbors=self.n_neighbors)


class K_Means():
    def __init__(self, features, labels, numb_clusters,seed=0, save_path=""):
        self.features = features
        self.labels = labels
        self.numb_clusters = numb_clusters
        self.seed = seed
        self.clf = KMeans(n_clusters=numb_clusters, random_state=seed)
        self.model_path = 'models/best_model_' + save_path + 'kmeans.sav'

    def load_pretrained(self):
        #Load a pretrained model to test
        self.clf = joblib.load(self.model_path)

    def train(self, idx_train):
        features_tr = self.features[idx_train]
        self.clusters = self.clf.fit_predict(features_tr)
        joblib.dump(self.clf, self.model_path)

    def classify(self, idx_test):
        self.features_test = self.features[idx_test]
        self.labels_test = self.labels[idx_test]
        self.prediction = self.clf.predict(self.features_test)

    def accuracy(self, classes):
        c_m = confusion_matrix(self.labels_test, self.prediction)
        for i in range(len(classes)):
            labels_count = np.sum(self.labels_test == i)
            c_m[i,:] = (c_m[i,:] /labels_count)*100
        return c_m, error_func(self.labels_test, self.prediction)

    def reset(self):
        self.clf = KMeans(n_clusters=self.numb_clusters, random_state=self.seed)

class Random_Forest():
    def __init__(self, features, labels, n_estimators, max_depth, criterion='gini', seed=0, save_path=""):
        self.seed = seed
        self.features = features
        self.labels = labels
        self.n_estimators = n_estimators
        self.n_estimators = max_depth
        self.clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,random_state=seed)
        self.model_path = 'models/best_model_' + save_path + 'rf.sav'

    def load_pretrained(self):
        #Load a pretrained model to test
        self.clf = joblib.load(self.model_path)

    def train(self, idx_train):
        features_tr = self.features[idx_train]
        labels_tr = self.labels[idx_train]
        self.clf.fit(features_tr,labels_tr)
        joblib.dump(self.clf, self.model_path)

    def classify(self, idx_test):
        self.features_test = self.features[idx_test]
        self.labels_test = self.labels[idx_test]
        self.prediction = self.clf.predict(self.features_test)

    def accuracy(self, classes):
        c_m = confusion_matrix(self.labels_test, self.prediction)
        for i in range(len(classes)):
            labels_count = np.sum(self.labels_test == i)
            c_m[i,:] = (c_m[i,:] /labels_count)*100
        return c_m, error_func(self.labels_test, self.prediction)


    def reset(self):
        self.clf = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.n_estimators,random_state=self.seed)


class MLP():
    def __init__(self, features, labels, seed=0, solver='adam', alpha=1e-5, hidden_layers=(25, 25), lr=1e-4, max_iter=1000, save_path=""):
        self.seed = seed
        self.features = features
        self.labels = labels
        self.solver = solver
        self.alpha = alpha
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.max_iter = max_iter

        self.clf = MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=hidden_layers,
                                 shuffle=True, max_iter=max_iter, learning_rate_init=lr, random_state=self.seed)

        self.model_path = 'models/best_model_' + save_path + 'mlp.sav'


    def load_pretrained(self):
        #Load a pretrained model to test
        self.clf = joblib.load(self.model_path)

    def train(self, idx_train):
        features_tr = self.features[idx_train]
        labels_tr = self.labels[idx_train]
        self.clf.fit(features_tr, labels_tr)
        joblib.dump(self.clf, self.model_path)

    def classify(self, idx_test):
        self.features_test = self.features[idx_test]
        self.labels_test = self.labels[idx_test]
        self.prediction = self.clf.predict(self.features_test)

    def accuracy(self, classes):
        c_m = confusion_matrix(self.labels_test, self.prediction)
        for i in range(len(classes)):
            labels_count = np.sum(self.labels_test == i)
            c_m[i,:] = (c_m[i,:] /labels_count)*100
        return c_m, error_func(self.labels_test, self.prediction)


    def reset(self):
        self.clf = MLPClassifier(solver=self.solver, alpha=self.alpha, hidden_layer_sizes=self.hidden_layers,
                                 shuffle=True, max_iter=self.max_iter, learning_rate_init=self.lr, random_state=self.seed)
