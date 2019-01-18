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

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32).copy()
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)).clone()
    values = torch.from_numpy(sparse_mx.data.copy()).clone()
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

class TrainerKHop():
    def __init__(self, model, adjacency, adjacency2, features, labels, cuda=True, regularization=None, lr=0.01, weight_decay=5e-4, batch_size=100, model_path=""):
        self.model = model
        self.adjacency = adjacency
        self.adjacency2 = adjacency2
        self.features = features
        self.labels = labels
        self.cuda = cuda
        self.regularization = regularization
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.model_path = model_path
        self.best_val_acc = 0

        if self.cuda:
            self.model = self.model.cuda()
            self.features = self.features.cuda()
            self.adjacency = self.adjacency.cuda()
            self.adjacency2 = self.adjacency2.cuda()
            self.labels = self.labels.cuda()

    def train(self, epoch, idx_train, idx_val):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.features, self.adjacency, self.adjacency2)
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

        self.model.eval()
        output = self.model(self.features, self.adjacency, self.adjacency2)
        loss_val = F.nll_loss(output[idx_val], self.labels[idx_val])
        acc_val = accuracy_prob(output[idx_val], self.labels[idx_val])

        if epoch % 25 == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train),
                  'acc_train: {:.4f}'.format(acc_train),
                  'loss_val: {:.4f}'.format(loss_val),
                  'acc_val: {:.4f}'.format(acc_val),
                  'time: {:.4f}s'.format(time.time() - t))

        if self.best_val_acc < acc_val.mean():
            self.best_val_acc = acc_val.mean()
            torch.save(self.model.state_dict(), self.model_path)

    def test(self, idx_test):
        self.model.eval()
        output = self.model(self.features, self.adjacency, self.adjacency2)
        test_output = output[idx_test]
        return test_output
