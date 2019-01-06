import numpy as np
import pdb
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
from models import GCN

class Trainer():
    def __init__(self, adjacency, features, labels, hidden, n_class, dropout=0.5, cuda=True,lr=0.01):
        self.adjacency = adjacency
        self.features = features
        self.labels = labels
        self.hidden = hidden
        self.n_class = n_class
        self.dropout = dropout
        self.cuda = cuda
        self.lr = lr
        self.model = GCN(nfeat=features.shape[1],
                    nhid=hidden,
                    nclass=n_class,
                    dropout=dropout)

        self.optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, weight_decay=args.weight_decay)

        if self.cuda:
            self.model.cuda()
            self.features = self.features.cuda()
            self.adjaceny = self.adjacency.cuda()
            self.labels = self.labels.cuda()

    def train(self, features, labels):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        self.output = self.model(self.features, self.adjaceny)

        loss_train.backward()
        self.optimizer.step()

    def test(self, features_test):
        pass