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

class Trainer():
    def __init__(self, model, adjacency, features, labels, cuda=True, regularization=None, lr=0.01, weight_decay=5e-4, batch_size=100, model_path=""):
        self.model = model
        self.adjacency = adjacency
        self.features = features
        self.labels = labels
        self.cuda = cuda
        self.regularization = regularization
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        if self.cuda:
            self.model = self.model.cuda()
            #self.features = self.features.cuda()
            #self.adjacency = self.adjacency.cuda()
            self.labels = self.labels.cuda()

    def train(self, epoch, idx_train, idx_val):
        loss_train = []
        loss_val = []
        acc_train = []
        acc_val = []
        best_val_acc = 0
        t = time.time()
        #Training with batches
        self.model.train()
        for b in range(0, idx_train.shape[0], self.batch_size):
            #print('BATCH: ', b)
            #import pdb
            #pdb.set_trace()
            mini_batch_size = self.batch_size
            if b == idx_train.shape[0]//self.batch_size:
                mini_batch_size = idx_train.shape[0] - b
            self.adj_part = self.adjacency[idx_train[b : b + mini_batch_size]][:, idx_train[b:b + mini_batch_size]]
            self.adj_part = sp.coo_matrix(self.adj_part)
            self.adj_part = sparse_mx_to_torch_sparse_tensor(self.adj_part)
            self.feat_part = self.features[idx_train[b : b + mini_batch_size]]
            if self.cuda:
                self.adj_part = self.adj_part.cuda()
                self.feat_part = self.feat_part.cuda()
            self.optimizer.zero_grad()

            output = self.model(self.feat_part, self.adj_part)
            regularization_loss = 0
            if self.regularization == 'l1':
                for param in self.model.parameters():
                    regularization_loss += torch.sum(torch.abs(param))

                loss_train.append(F.nll_loss(output, self.labels[idx_train[b : b + mini_batch_size]]) + 0.001*regularization_loss)
            else:
                loss_train.append(F.nll_loss(output, self.labels[idx_train[b : b + mini_batch_size]]))
            acc_train.append(accuracy_prob(output, self.labels[idx_train[b : b + mini_batch_size]]))
            loss_train[-1].backward()
            self.optimizer.step()

        self.model.eval()
        for b in range(0, idx_val.shape[0], self.batch_size):
            mini_batch_size = self.batch_size
            if b == idx_val.shape[0] // self.batch_size:
                mini_batch_size = idx_val.shape[0] - b
            #Validation
            self.adj_part = self.adjacency[idx_val[b : b + mini_batch_size]][:, idx_val[b:b + mini_batch_size]]
            self.adj_part = sp.coo_matrix(self.adj_part)
            self.adj_part = sparse_mx_to_torch_sparse_tensor(self.adj_part)
            self.feat_part = self.features[idx_val[b : b + mini_batch_size]]
            if self.cuda:
                self.adj_part = self.adj_part.cuda()
                self.feat_part = self.feat_part.cuda()
            output = self.model(self.feat_part, self.adj_part)
            loss_val.append(F.nll_loss(output, self.labels[idx_val[b : b + mini_batch_size]]))
            acc_val.append(accuracy_prob(output, self.labels[idx_val[b : b + mini_batch_size]]))

        loss_train = torch.FloatTensor(loss_train)
        loss_val = torch.FloatTensor(loss_val)
        acc_train = torch.FloatTensor(acc_train)
        acc_val = torch.FloatTensor(acc_val)
        if epoch % 25 == 0:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss_train.mean()),
                  'acc_train: {:.4f}'.format(acc_train.mean()),
                  'loss_val: {:.4f}'.format(loss_val.mean()),
                  'acc_val: {:.4f}'.format(acc_val.mean()),
                  'time: {:.4f}s'.format(time.time() - t))

        if best_val_acc < acc_val.mean():
            best_val_acc = acc_val.mean()
            torch.save(self.model.state_dict(), model_path)

    def test(self, idx_test):
        self.model.eval()
        test_output = None
        for b in range(0, idx_test.shape[0], self.batch_size):
            mini_batch_size = self.batch_size
            if b == idx_test.shape[0] // self.batch_size:
                mini_batch_size = idx_test.shape[0] - b
            self.adj_part = self.adjacency[idx_test[b : b + mini_batch_size]][:, idx_test[b:b + mini_batch_size]]
            self.adj_part = sp.coo_matrix(self.adj_part)
            self.adj_part = sparse_mx_to_torch_sparse_tensor(self.adj_part)
            self.feat_part = self.features[idx_test[b : b + mini_batch_size]]
            if self.cuda:
                self.adj_part = self.adj_part.cuda()
                self.feat_part = self.feat_part.cuda()
            output = self.model(self.feat_part, self.adj_part)
            if test_output is not None:
                test_output = torch.cat((test_output, output))
            else:
                test_output = output.clone()
        return test_output
