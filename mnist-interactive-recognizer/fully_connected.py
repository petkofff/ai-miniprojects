import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class FullyConnected(nn.Module):
    MNIST_IM_SHAPE = (28, 28)

    def __init__(self):
        super(FullyConnected, self).__init__()
        self.__input_size = self.MNIST_IM_SHAPE[0] * self.MNIST_IM_SHAPE[1]
        self.__layer_sizes = [self.__input_size, 1000, 100, 10]

        self.loss_fn = nn.CrossEntropyLoss()

        self.fc1 = nn.Linear(self.__layer_sizes[0], self.__layer_sizes[1])
        self.fc2 = nn.Linear(self.__layer_sizes[1], self.__layer_sizes[2])
        self.fc3 = nn.Linear(self.__layer_sizes[2], self.__layer_sizes[3])

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
