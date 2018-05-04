import torch
import torchvision
import torchvision.transforms as transforms


import argparse

parser = argparse.ArgumentParser(description='Bifurcate NN.')
parser.add_argument('--mode', help='mode, default to bifurcate', default="bifurcate")
parser.add_argument('--penalty', help='penalty type', default="l1loss", choices=list(["l1loss","l2loss"]))
parser.add_argument('--penalty-factor', type=float, default=0.1, help='penalty factor')
parser.add_argument('--penalty-decay', type=float, default=0.9, help='penalty decay')

args = parser.parse_args()

nettype = args.mode
penalty_factor = args.penalty_factor
penalty_decay = args.penalty_decay
penalty_type = args.penalty

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import matplotlib.pyplot as plt
import numpy as np

# functions to show an image

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
plt.show()
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

import torch.nn as nn
import torch.nn.functional as F

print("mode: %s" % nettype)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        if nettype == "bifurcate":
            self.conv1_1 = nn.Conv2d(3, 12, 5)
            self.pool_1 = nn.MaxPool2d(2, 2)
            self.conv2_1 = nn.Conv2d(12, 16, 5)
            self.fc1_1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2_1 = nn.Linear(120, 84)
            self.fc3_1 = nn.Linear(84, 10)

            self.conv1_2 = nn.Conv2d(3, 12, 5)
            self.pool_2 = nn.MaxPool2d(2, 2)
            self.conv2_2 = nn.Conv2d(12, 16, 5)
            self.fc1_2 = nn.Linear(16 * 5 * 5, 120)
            self.fc2_2 = nn.Linear(120, 84)
            self.fc3_2 = nn.Linear(84, 10)
        else:
            self.conv1 = nn.Conv2d(3, 12, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(12, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

    def forward(self, x0):
        if nettype == "bifurcate":
            x1 = self.pool_1(F.relu(self.conv1_1(x0)))
            x1 = self.pool_1(F.relu(self.conv2_1(x1)))
            x1 = x1.view(-1, 16 * 5 * 5)
            x1 = F.relu(self.fc1_1(x1))
            x1 = F.relu(self.fc2_1(x1))
            x1 = self.fc3_1(x1)

            x2 = self.pool_2(F.relu(self.conv1_2(x0)))
            x2 = self.pool_2(F.relu(self.conv2_2(x2)))
            x2 = x2.view(-1, 16 * 5 * 5)
            x2 = F.relu(self.fc1_2(x2))
            x2 = F.relu(self.fc2_2(x2))
            x2 = self.fc3_2(x2)
            x_final = x1 * x2
        else:
            x = self.pool(F.relu(self.conv1(x0)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16 * 5 * 5)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x_final = self.fc3(x)

        return x_final


net = Net()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assume that we are on a CUDA machine, then this should print a CUDA device:

print(device)

net.to(device)

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
if penalty_type == "l1loss":
    penalty = nn.L1Loss()
if penalty_type == "l2loss":
    penalty = nn.MSELoss()

print("penalty factor: %f" % penalty_factor)
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(8):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        if nettype == "bifurcate":
            loss += penalty_factor / penalty(net.conv1_1.weight.data, net.conv1_2.weight.data)
            loss += penalty_factor / penalty(net.conv2_1.weight.data, net.conv2_2.weight.data)
            loss += penalty_factor / penalty(net.fc1_1.weight.data, net.fc1_2.weight.data)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%2d, %5d] loss: %.3f, pf: %5f' %
                  (epoch + 1, i + 1, running_loss / 2000, penalty_factor))
            penalty_factor *= penalty_decay
            running_loss = 0.0

print('Finished Training')


dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

outputs = net(images.to(device))

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.to("cpu") == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images.to(device))
        _, predicted = torch.max(outputs, 1)
        c = (predicted.to("cpu") == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))