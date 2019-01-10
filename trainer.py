import numpy as np
import pdb
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
import scipy.sparse as sp
from models import GCN
from error import accuracy_prob
from graph_analysis import Our_Graph



class Trainer():
    def __init__(self, adjacency, features, D_norm, labels, hidden, n_class, dropout=0.5, cuda=True,lr=0.01, weight_decay=5e-4, epochs=100):
        self.adjacency = adjacency
        self.features = features
        self.D_norm = D_norm
        self.labels = labels
        self.hidden = hidden
        self.n_class = n_class
        self.dropout = dropout
        self.cuda = cuda
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        idx_train = range(140)
        idx_val = range(200, 500)
        idx_test = range(500, 1500)
        self.model = GCN(nfeat=features.shape[1],
                    nhid=hidden,
                    nclass=n_class,
                    dropout=dropout)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.features = torch.FloatTensor(np.array(self.features.todense()))
        self.adjacency = sp.coo_matrix(self.adjacency)
        self.adjacency = self.D_norm @ (self.adjacency + sp.eye(self.adjacency.shape[0])) @ D_norm
        self.adjacency = sparse_mx_to_torch_sparse_tensor(self.adjacency)
        #!!!!!!!! np.where
        self.labels = torch.LongTensor(np.where(self.labels)[1])
        self.idx_train = torch.LongTensor(idx_train)
        self.idx_val = torch.LongTensor(idx_val)
        self.idx_test = torch.LongTensor(idx_test)

        if self.cuda:
            self.model.cuda()
            self.features = self.features.cuda()
            self.adjaceny = self.adjacency.cuda()
            self.labels = self.labels.cuda()

    def sparse_mx_to_torch_sparse_tensor(sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def train(self, epoch):
        t = time.time()
        #Training
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.features, self.adjaceny)
        acc_train = error_func(self.labels, self.output)
        loss_train = F.nll_loss(output[idx_train], self.labels[idx_train])
        acc_train = accuracy_prob(output[idx_train], self.labels[idx_train])
        loss_train.backward()
        self.optimizer.step()

        #Validation
        self.model.eval()
        output = self.model(self.features, self.adjaceny)
        loss_val = F.nll_loss(output[idx_val], self.labels[idx_val])
        acc_val = accuracy_prob(output[idx_val], self.labels[idx_val])

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

    def test(self):
        self.model.eval()
        output = self.model(self.features, self.adjaceny)
        loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        acc_test = accuracy_prob(output[idx_test], self.labels[idx_test])
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

    def run(self):
        for epoch in range(self.epochs):
            self.train(epoch)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))
        self.test()
