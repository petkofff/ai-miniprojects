import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms


class Convolutional(nn.Module):
    MNIST_IM_SHAPE = (28, 28)

    def __init__(self):
        super(Convolutional, self).__init__()

        self.__conv_dims = [1, 32, 64]

        self.conv1 = nn.Conv2d(self.__conv_dims[0],
                               self.__conv_dims[1],
                               kernel_size=5,
                               stride=1,
                               padding=2)

        self.conv2 = nn.Conv2d(self.__conv_dims[1],
                               self.__conv_dims[2],
                               kernel_size=5,
                               stride=1,
                               padding=2)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.drop_out = nn.Dropout()

        self.__linear_dims = [
            self.__conv_dims[-1] * ((self.MNIST_IM_SHAPE[0] // 4)**2), 1000, 10
        ]

        self.fc1 = nn.Linear(self.__linear_dims[0], self.__linear_dims[1])
        self.fc2 = nn.Linear(self.__linear_dims[1], self.__linear_dims[2])

        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape(-1, 1, self.MNIST_IM_SHAPE[0], self.MNIST_IM_SHAPE[1])

        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.relu(x)

        x = x.reshape(x.size(0), -1)
        x = self.drop_out(x)

        x = self.fc1(x)
        x = self.fc2(x)

        return x
