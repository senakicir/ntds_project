import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pdb
from error import accuracy_prob, error_func
from sklearn.metrics import confusion_matrix
# Hyper Parameters
# hidden_size = 500
# num_epochs = 10
# batch_size = 100
# learning_rate = 0.0001

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out_rep = self.relu(out)
        out = self.fc2(out_rep)
        return out, out_rep



class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data.copy()).float()
        self.target = torch.from_numpy(target.copy()).long()
        self.transform = transform

    def __getitem__(self, index):
       # pdb.set_trace()
        x = self.data[index]
        y = self.target[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

class MLP():
    def __init__(self,hidden_size, features, labels,num_epoch,batch_size,num_classes,lr=0.0001, save_path="",cuda=True,seed=0):
        self.features = features
        self.labels = labels.astype(np.int32)
        self.num_classes = num_classes
        self.seed = seed
        self.num_epochs = num_epoch
        self.batch_size = batch_size
        self.cuda = cuda
        self.lr = lr
        self.model_path = 'models/best_model_' + save_path + 'mlpnn.sav'
        input_size = self.features.shape[1]

        self.net = Net(input_size, hidden_size, self.num_classes)
        if self.cuda:
            self.net.cuda()

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
    def train(self, idx_train):
        self.net.train()
        dataset = MyDataset(self.features[idx_train], self.labels[idx_train])
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(data_loader):
                # Convert torch tensor to Variable
                if self.cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                # Forward + Backward + Optimize
                self.optimizer.zero_grad()  # zero the gradient buffer
                outputs, _ = self.net(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                if (i+1) % 100 == 0:
                    print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                           %(epoch+1, self.num_epochs, i+1, len(dataset)//self.batch_size, loss.item()))



    def save_model(self):
        torch.save(self.net.state_dict(), self.model_path)
    def load_pretrained(self):
        self.net.load_state_dict(torch.load(self.model_path))
    def classify(self, idx_test):
        correct = 0
        total = 0
        self.net.eval()
        self.labels_test = self.labels[idx_test]
        dataset = MyDataset(self.features[idx_test], self.labels_test)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False)
        self.prediction = np.array([])
        for images, labels in data_loader:
            if self.cuda:
                images = images.cuda()
            outputs, _ = self.net(images)
            _, prediction = torch.max(outputs.data, 1)
            self.prediction=np.concatenate([self.prediction,prediction.cpu().detach().numpy()])
            total += labels.size(0)
            correct += (prediction.cpu() == labels).sum()

        #print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    def accuracy(self, classes):
        c_m = confusion_matrix(self.labels_test, self.prediction)
        acc_test = error_func(self.prediction, self.labels_test)
        for i in range(len(classes)):
            labels_count = np.sum(self.labels_test == i)
            c_m[i,:] = (c_m[i,:] /labels_count)*100
        return c_m, acc_test

    def get_rep(self,features):
        self.load_pretrained()
        self.net.eval()
        dataset = MyDataset(features, np.zeros(features.shape[0]))
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False)
        new_rep = None
        for images, labels in data_loader:
            if self.cuda:
                images = images.cuda()
            _, rep = self.net(images)
            if new_rep is None:
                new_rep = rep.cpu().detach().numpy()
            else:
                new_rep = np.concatenate([new_rep,rep.cpu().detach().numpy()])

        return new_rep

    def reset(self):
        dict_model = self.net.state_dict()
        for k, v in dict_model.items():
            if "layer" in k:
                dict_model[k].reset_parameters()
