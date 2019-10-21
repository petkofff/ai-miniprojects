# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mnist_im_shape = (28, 28)
input_size = mnist_im_shape[0]*mnist_im_shape[1]

# hyperparameters
layer_sizes = [input_size, 600, 10]
num_epochs = 5
batch_size = 200
learning_rate = 0.0002

train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

model = nn.Sequential(
        nn.Linear(layer_sizes[0], layer_sizes[1]),
        nn.ReLU(),
        nn.Linear(layer_sizes[1], layer_sizes[2]),
        )

def train(model):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        print('epoch: [{}/{}]\n'.format(epoch+1, num_epochs))

        for i, (images, labels) in enumerate(train_loader):

            # prepare
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print('loss: {}'.format(loss.item()))


def test(model):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 28*28)
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted==labels).sum().item()

        print('accuracy: {} %'.format(100 * correct / total))

train(model)
test(model)

from screen import *

ds = DrawingScreen(10, 14)

def handler(scr):
    with torch.no_grad():
        tensor_im = torch.Tensor(scr.reshape(-1, 28*28))
        output = model(tensor_im)
        prediction = np.array(output[0]).argmax()
        print('prediction: {}'.format(prediction))

ds.mainloop(handler)
