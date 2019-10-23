import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

class FullyConnected(nn.Module):
    MNIST_IM_SHAPE = (28, 28)

    def __init__(self):
        super(FullyConnected, self).__init__()
        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__input_size = self.MNIST_IM_SHAPE[0]*self.MNIST_IM_SHAPE[1]
        self.__layer_sizes = [self.__input_size, 1000, 100, 10]

        self.loss_fn = nn.CrossEntropyLoss()

        self.fc1 = nn.Linear(self.__layer_sizes[0], self.__layer_sizes[1])
        self.fc2 = nn.Linear(self.__layer_sizes[1], self.__layer_sizes[2])
        self.fc3 = nn.Linear(self.__layer_sizes[2], self.__layer_sizes[3])

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    def train(self, train_loader, num_epochs = 5, learning_rate = 0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            print('epoch: [{}/{}]\n'.format(epoch+1, num_epochs))

            for i, (images, labels) in enumerate(train_loader):

                # prepare
                images = images.reshape(-1, self.__input_size).to(self.__device)
                labels = labels.to(self.__device)

                # forward
                outputs = self(images)
                loss = self.loss_fn(outputs, labels)

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (i+1) % 10 == 0:
                    print('loss: {}'.format(loss.item()))

    def test(self, test_loader):
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = images.reshape(-1, 28*28)
                output = self(images)
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted==labels).sum().item()

            accuracy = 100*correct/total
            print('accuracy: {} %'.format(accuracy))
            return accuracy


# if __name__ == '__main__':
#     train_dataset = torchvision.datasets.MNIST(root='../../data',
#                                             train=True,
#                                             transform=transforms.ToTensor(),
#                                             download=True)

#     test_dataset = torchvision.datasets.MNIST(root='../../data',
#                                             train=False,
#                                             transform=transforms.ToTensor())

#     train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                             batch_size=200,
#                                             shuffle=True)

#     test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                             batch_size=200,
#                                             shuffle=False)

#     model = FullyConnected()
#     model.train(train_loader)
#     model.test(test_loader)
