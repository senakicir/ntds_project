import numpy as np
import pdb
import math
# import torch
# from torch.nn.parameter import Parameter
# from torch.nn.modules.module import Module
from sklearn.svm import SVC

# class GraphConvolution(Module):
#     """
#     Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
#     """
#
#     def __init__(self, in_features, out_features, bias=True):
#         super(GraphConvolution, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.weight = Parameter(torch.FloatTensor(in_features, out_features))
#         if bias:
#             self.bias = Parameter(torch.FloatTensor(out_features))
#         else:
#             self.register_parameter('bias', None)
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.weight.size(1))
#         self.weight.data.uniform_(-stdv, stdv)
#         if self.bias is not None:
#             self.bias.data.uniform_(-stdv, stdv)
#
#     def forward(self, input, adj):
#         support = torch.mm(input, self.weight)
#         output = torch.spmm(adj, support)
#         if self.bias is not None:
#             return output + self.bias
#         else:
#             return output
#
#     def __repr__(self):
#         return self.__class__.__name__ + ' (' \
#                + str(self.in_features) + ' -> ' \
#                + str(self.out_features) + ')'

class Graph_NN():
    def __init(self):
        pass

class SVM():
    def __init__(self):
        self.clf = SVC(gamma='auto')

    def train(self, features_tr, labels_tr):
        self.clf.fit(features_tr, labels_tr)

    def classify(self, features_test):
        return self.clf.predict(features_test)

class K_Means():
    def __init__(self):
        pass

class Random_Forest():
    def __init__(self):
        pass
