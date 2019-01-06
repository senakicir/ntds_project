import numpy as np
import pdb
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
from models import GCN
from error import error_func

class Trainer():
    def __init__(self, adjacency, features, labels, hidden, n_class, dropout=0.5, cuda=True,lr=0.01, weight_decay=5e-4, epochs=10):
        self.adjacency = adjacency
        self.features = features
        self.labels = labels
        self.hidden = hidden
        self.n_class = n_class
        self.dropout = dropout
        self.cuda = cuda
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.model = GCN(nfeat=features.shape[1],
                    nhid=hidden,
                    nclass=n_class,
                    dropout=dropout)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        if self.cuda:
            self.model.cuda()
            self.features = self.features.cuda()
            self.adjaceny = self.adjacency.cuda()
            self.labels = self.labels.cuda()

    def train(self, epoch):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        self.output = self.model(self.features, self.adjaceny)
        acc_train = error_func(self.labels, self.output)
        loss_train = F.nll_loss(output, labels)
        loss_train.backward()
        self.optimizer.step()

        self.model.eval()
        self.output = self.model(self.features, self.adjaceny)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_train = error_func(self.labels, self.output)

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

    def test(self, epoch):
        self.model.eval()
        output = model(features, adj)
        pass

    def run(self):
        for epoch in range(self.epochs):
            train(epoch)
            test(epoch)