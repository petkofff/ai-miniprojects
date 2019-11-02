import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from fully_connected import *
from convolutional import *
from screen import *

batch_size = 300

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


def train(model,
          train_loader,
          num_epochs=5,
          learning_rate=0.001,
          device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        print('epoch: [{}/{}]\n'.format(epoch + 1, num_epochs))

        for i, (images, labels) in enumerate(train_loader):

            # prepare
            images = images.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 10 == 0:
                print('loss: {}'.format(loss.item()))


def test(model, test_loader):
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            output = model(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print('accuracy: {} %'.format(accuracy))
        return accuracy


model = Convolutional()

train(model, train_loader)
test(model, test_loader)

ds = DrawingScreen(10, 14)


def handler(scr):
    with torch.no_grad():
        tensor_im = torch.Tensor(scr)
        # print(tensor_im)
        output = model(tensor_im)
        prediction = np.array(output[0]).argmax()
        print('prediction: {}'.format(prediction))


ds.mainloop(handler)
