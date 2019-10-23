import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

batch_size = 200

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
from fully_connected import *

model = FullyConnected()
model.train(train_loader)
model.test(test_loader)

from screen import *

ds = DrawingScreen(10, 14)

def handler(scr):
    with torch.no_grad():
        tensor_im = torch.Tensor(scr.reshape(-1, 28*28))
        output = model(tensor_im)
        prediction = np.array(output[0]).argmax()
        print('prediction: {}'.format(prediction))

ds.mainloop(handler)
