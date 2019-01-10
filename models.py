import numpy as np
import pdb
import math
import scipy.sparse as sp
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from trainer import Trainer


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

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

class GCN():
    def __init__(self, nhid, dropout, adjacency, features, D_norm, labels, cuda=True, lr=0.01, weight_decay=5e-4, epochs=100):
        self.adjacency = adjacency
        self.features = features
        self.D_norm = D_norm
        self.labels = labels
        self.nfeat = features.shape[-1]
        self.nhid = nhid
        self.nclass = labels.shape[-1]
        self.epochs = epochs
        self.gcn = GraphNeuralNet(self.nfeat, self.nhid, self.nclass, dropout)
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
        self.features = torch.FloatTensor(np.array(self.features))
        self.adjacency = sp.coo_matrix(self.adjacency)
        self.adjacency = self.D_norm @ (self.adjacency + sp.eye(self.adjacency.shape[0])) @ D_norm
        self.adjacency = sparse_mx_to_torch_sparse_tensor(self.adjacency)
        self.labels = torch.LongTensor(np.where(self.labels)[1])
        self.idx_train = torch.LongTensor(idx_train)
        self.idx_val = torch.LongTensor(idx_val)
        self.idx_test = torch.LongTensor(idx_test)

        #Create trainer
        self.trainer = Trainer(self.gcn, self.adjacency, self.features, self.labels, cuda, lr, weight_decay)

    def train(self, idx_train, idx_test):
        for epoch in range(self.epochs):
            self.trainer.train(epoch, idx_train, idx_test)

    def classify(self, idx_test):
        self.trainer.test(idx_test)

class SVM():
    def __init__(self,kernel,poly_degree=3,seed=0):
        if kernel == 'linear':
            self.clf = svm.LinearSVC(multi_class='ovr',random_state=seed) # Can use 'crammer_singer’ but more expensive while not that much better accuracy(only more stable)
        else:
            self.clf = svm.SVC(gamma='auto', kernel=kernel,degree=poly_degree,decision_function_shape='ovr',random_state=seed)

    def train(self, features_tr, labels_tr):
        self.clf.fit(features_tr, labels_tr)

    def classify(self, features_test):
        return self.clf.predict(features_test)

    def accuracy(self, features_test, labels_test):
        return self.clf.score(features_test,labels_test)

class KNN():
    def __init__(self):
        self.clf = KNeighborsClassifier(n_neighbors=42)

    def train(self, features_tr, labels_tr):
        self.clf.fit(features_tr, labels_tr)

    def classify(self, features_test):
        return self.clf.predict(features_test)

    def reset(self, new_n_neigh):
        self.clf = KNeighborsClassifier(n_neighbors=new_n_neigh)


class K_Means():
    def __init__(self, numb_clusters,seed=0):
        self.clf = KMeans(n_clusters=numb_clusters, random_state=seed)

    def train(self, features_tr,labels_tr):
        self.clusters = self.clf.fit_predict(features_tr)

    def classify(self, features_test):
        return self.clf.predict(features_test)

class Random_Forest():
    def __init__(self,n_estimators, max_depth, criterion='gini', seed=0):
        self.seed = seed
        self.clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,random_state=seed)

    def train(self, features_tr, labels_tr):
        self.clf.fit(features_tr,labels_tr)
    def classify(self, features_test):
        return self.clf.predict(features_test)
    def accuracy(self, features_test, labels_test):
        return self.clf.score(features_test,labels_test)
    def reset(self, new_n_est, new_max_depth):
        self.clf = RandomForestClassifier(n_estimators=new_n_est, max_depth=new_max_depth,random_state=self.seed)
