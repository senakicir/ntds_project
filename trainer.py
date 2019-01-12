import numpy as np
import pdb
import math
import time
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.optim as optim
import scipy.sparse as sp
from error import accuracy_prob
from graph_analysis import Our_Graph



class Trainer():
    def __init__(self, model, adjacency, features, labels, cuda=True, regularization=None, lr=0.01, weight_decay=5e-4):
        self.model = model
        self.adjacency = adjacency
        self.features = features
        self.labels = labels
        self.cuda = cuda
        self.regularization = regularization
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        if self.cuda:
            self.model.cuda()
            self.features = self.features.cuda()
            self.adjacency = self.adjacency.cuda()
            self.labels = self.labels.cuda()

    def train(self, epoch, idx_train, idx_val):
        t = time.time()
        #Training
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.features, self.adjacency)
        regularization_loss = 0
        if self.regularization == 'l1':
            for param in self.model.parameters():
                regularization_loss += torch.sum(torch.abs(param))

            loss_train = F.nll_loss(output[idx_train], self.labels[idx_train]) + 0.001*regularization_loss
        else:
            loss_train = F.nll_loss(output[idx_train], self.labels[idx_train])
        acc_train = accuracy_prob(output[idx_train], self.labels[idx_train])
        loss_train.backward()
        self.optimizer.step()

        #Validation
        self.model.eval()
        output = self.model(self.features, self.adjacency)
        loss_val = F.nll_loss(output[idx_val], self.labels[idx_val])
        acc_val = accuracy_prob(output[idx_val], self.labels[idx_val])

        if epoch % 25 == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'acc_train: {:.4f}'.format(acc_train.item()),
                  'loss_val: {:.4f}'.format(loss_val.item()),
                  'acc_val: {:.4f}'.format(acc_val.item()),
                  'time: {:.4f}s'.format(time.time() - t))

    def test(self, idx_test):
        self.model.eval()
        output = self.model(self.features, self.adjacency)
        #loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        return output[idx_test]
