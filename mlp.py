import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms


# Hyper Parameters
hidden_size = 500
num_epochs = 10
batch_size = 100
learning_rate = 0.0001

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

from torch.utils.data import Dataset, DataLoader

import numpy as np
import pdb

class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = torch.from_numpy(data).float()
        self.target = torch.from_numpy(target).long()
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

def do_mlp(x_train, y_train,x_test, y_test,num_classes):
    # MNIST Dataset
    # train_dataset = dsets.MNIST(root='../data',
    #                             train=True,
    #                             transform=transforms.ToTensor(),
    #                             download=True)
    #
    # test_dataset = dsets.MNIST(root='../data',
    #                            train=False,
    #                            transform=transforms.ToTensor())
    input_size = x_train.shape[1]
    pdb.set_trace()
    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)
    pdb.set_trace()
    train_dataset = MyDataset(x_train, y_train)

    test_dataset = MyDataset(x_test, y_test)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # Neural Network Model (1 hidden layer)


    net = Net(input_size, hidden_size, num_classes)
    net.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        for i, (x, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            images = x.cuda()
            labels = labels.cuda()

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f'
                       %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.item()))

    # Test the Model
    correct = 0
    total = 0
    for x, labels in test_loader:
        images = x.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    # Save the Model
    torch.save(net.state_dict(), 'model.pkl')

