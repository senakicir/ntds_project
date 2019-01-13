import numpy as np
import pdb
import math
import scipy.sparse as sp
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torch.nn import BatchNorm1d
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from trainer import Trainer
from error import accuracy_prob, error_func


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

    def forward(self, x, adj):
        x = F.relu(self.bn1(self.gc1(x, adj)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.bn2(self.gc2(x, adj)))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc3(x, adj)
        return F.log_softmax(x, dim=1)

class GCN():
    def __init__(self, nhid, dropout, adjacency, features, labels, cuda=True, regularization=None, lr=0.01, weight_decay=5e-4, epochs=100):
        self.adjacency = adjacency
        self.features = features
        self.labels = labels
        self.nfeat = features.shape[-1]
        self.nhid = nhid
        self.nclass = labels.shape[-1]
        self.epochs = epochs
        self.gcn = GraphNeuralNet(self.nfeat, self.nhid, self.nclass, dropout)

        #Train-val-test split
        #idxs = np.random.permutation(np.arange(features.shape[0]))
        #idx_train = idxs[:int(features.shape[0]*0.60)]
        #idx_val = idxs[int(features.shape[0]*0.60): int(features.shape[0]*0.60) + int(features.shape[0]*0.20)]
        #idx_test = idxs[int(features.shape[0]*0.60) + int(features.shape[0]*0.20):]

        self.features = torch.FloatTensor(np.array(self.features))
        self.adjacency = (self.adjacency + np.eye(self.adjacency.shape[0]))
        self.D = np.diag(self.adjacency.sum(axis=1))
        self.D_norm = (np.power(np.linalg.inv(self.D), 0.5))
        self.adjacency = self.D_norm @ (self.adjacency) @ self.D_norm
        self.adjacency = sp.coo_matrix(self.adjacency)
        self.adjacency = sparse_mx_to_torch_sparse_tensor(self.adjacency)
        self.labels = torch.LongTensor(np.where(self.labels)[1])

        #Create trainer
        self.trainer = Trainer(self.gcn, self.adjacency, self.features, self.labels, cuda, regularization, lr, weight_decay)

    def train(self, idx_train):
        train_size = idx_train.shape[0]
        valid_size = train_size // 5
        idx_val = np.random.permutation(np.arange(train_size))[0:valid_size]
        self.idx_train = torch.LongTensor(idx_train)
        self.idx_val = torch.LongTensor(idx_val)
        for epoch in range(self.epochs):
            self.trainer.train(epoch, self.idx_train, self.idx_val)

    def classify(self, idx_test):
        self.labels_test = self.labels[idx_test]
        self.prediction = self.trainer.test(idx_test)

    def accuracy(self):
        #confusion_matrix(self.labels_test, self.prediction)
        acc_test = accuracy_prob(self.prediction, self.labels_test)
        return acc_test

class SVM():
    def __init__(self, features, labels, kernel,poly_degree=3,seed=0):
        self.features = features
        self.labels = labels
        if kernel == 'linear':
            self.clf = svm.LinearSVC(multi_class='ovr',random_state=seed) # Can use 'crammer_singer’ but more expensive while not that much better accuracy(only more stable)
        else:
            self.clf = svm.SVC(gamma='auto', kernel=kernel,degree=poly_degree,decision_function_shape='ovr',random_state=seed)

    def train(self, idx_train):
        features_tr = self.features[idx_train]
        labels_tr = self.labels[idx_train]
        self.clf.fit(features_tr, labels_tr)

    def classify(self, idx_test):
        self.features_test = self.features[idx_test]
        self.labels_test = self.labels[idx_test]
        self.prediction = self.clf.predict(self.features_test)

    def accuracy(self):
        #confusion_matrix(self.labels_test, self.prediction)
        return error_func(self.labels_test, self.prediction)
#        return self.clf.score(self.features_test,self.labels_test)

class KNN():
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.clf = KNeighborsClassifier(n_neighbors=42)

    def train(self, idx_train):
        features_tr = self.features[idx_train]
        labels_tr = self.labels[idx_train]
        self.clf.fit(features_tr, labels_tr)

    def reset(self, new_n_neigh):
        self.clf = KNeighborsClassifier(n_neighbors=new_n_neigh)

    def classify(self, idx_test):
        self.features_test = self.features[idx_test]
        self.labels_test = self.labels[idx_test]
        self.prediction = self.clf.predict(self.features_test)

    def accuracy(self):
        #confusion_matrix(self.labels_test, self.prediction)
        return error_func(self.labels_test, self.prediction)


class K_Means():
    def __init__(self, features, labels, numb_clusters,seed=0):
        self.features = features
        self.labels = labels
        self.clf = KMeans(n_clusters=numb_clusters, random_state=seed)

    def train(self, idx_train):
        features_tr = self.features[idx_train]
        self.clusters = self.clf.fit_predict(features_tr)

    def classify(self, idx_test):
        self.features_test = self.features[idx_test]
        self.labels_test = self.labels[idx_test]
        self.prediction = self.clf.predict(self.features_test)

    def accuracy(self):
        #confusion_matrix(self.labels_test, self.prediction)
        return error_func(self.labels_test, self.prediction)

class Random_Forest():
    def __init__(self, features, labels, n_estimators, max_depth, criterion='gini', seed=0):
        self.seed = seed
        self.features = features
        self.labels = labels
        self.clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,random_state=seed)

    def train(self, idx_train):
        features_tr = self.features[idx_train]
        labels_tr = self.labels[idx_train]
        self.clf.fit(features_tr,labels_tr)

    def classify(self, idx_test):
        self.features_test = self.features[idx_test]
        self.labels_test = self.labels[idx_test]
        self.prediction = self.clf.predict(self.features_test)

    def accuracy(self):
        #confusion_matrix(self.labels_test, self.prediction)
        return error_func(self.labels_test, self.prediction)

    def reset(self, new_n_est, new_max_depth):
        self.clf = RandomForestClassifier(n_estimators=new_n_est, max_depth=new_max_depth,random_state=self.seed)


class MLP():
    def __init__(self, features, labels, seed=0, solver='adam', alpha=1e-5, hidden_layers=(25, 25), lr=1e-4, max_iter=1000):
        self.seed = seed
        self.features = features
        self.labels = labels
        self.clf = MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=hidden_layers,
                                 shuffle=True, max_iter=max_iter, learning_rate_init=lr, random_state=self.seed)

    def train(self, idx_train):
        features_tr = self.features[idx_train]
        labels_tr = self.labels[idx_train]
        self.clf.fit(features_tr, labels_tr)

    def classify(self, idx_test):
        self.features_test = self.features[idx_test]
        self.labels_test = self.labels[idx_test]
        self.prediction = self.clf.predict(self.features_test)

    def accuracy(self):
        return error_func(self.labels_test, self.prediction)

    def reset(self, solver='adam', alpha=1e-5, hidden_layers=(25, 25), lr=1e-4, max_iter=5000):
        self.clf = MLPClassifier(solver=solver, alpha=alpha, hidden_layer_sizes=hidden_layers,
                                 shuffle=True, max_iter=max_iter, learning_rate_init=lr, random_state=self.seed)
