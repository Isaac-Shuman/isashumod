#change to master

#import statements
import torch
import torchvision
import torchvision.transforms.v2 as transforms
from dataLoader import CustomImageDataset #import dataloader file
import os
import time
from IPython import embed
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import _warnings
from matplotlib import pylab as plt
from math import floor
import logging
from torch.utils.data import random_split
#Setup for the run info files
def setupInfoFiles():
    rootFolderName = '/mnt/tmpdata/data/isashu/runFolders'
    folderName = os.path.join(rootFolderName,str(time.strftime('run_%Y-%m-%d_%H_%M_%S')))
    logName = 'metrics.log'

    os.mkdir(folderName)
    log_filename = os.path.join(folderName, logName)

    pathToModel = os.path.join(folderName, 'model.pth')

    return pathToModel, log_filename

#Used to calculate the dimensions of an image after it passed through a pool or convolution layer
def calcLengOfConv(length , kern_size, stride_size = -1):
    if stride_size == -1:
        stride_size = kern_size
    return floor((length - kern_size)/stride_size + 1)

#Make the data loaders
def make_datasets(rootForTrain, rootForTest):

    iiFile = "imageNameAndImageDS.hdf5"
    isFile = "imageNameAndSpots.csv"
    hd_filename = os.path.join(rootForTrain, iiFile)
    cs_filename = os.path.join(rootForTrain, isFile)
    trainset = CustomImageDataset(annotations_file=cs_filename, path_to_hdf5=hd_filename, transform=None)

    hd_filename = os.path.join(rootForTest, iiFile)
    cs_filename = os.path.join(rootForTest, isFile)
    testset = CustomImageDataset(annotations_file=cs_filename, path_to_hdf5=hd_filename, transform=None)

    return trainset, testset

def make_dataloaders(trainset, testset,batch_size):
    #Split the training set into 50% training and 50% validation data


    test_abs = int(len(trainset) * 0.5)
    generator = torch.Generator().manual_seed(1)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs], generator=generator
    )
    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size,
                                             shuffle=True)
    valloader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size,
                                             shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)
    return trainloader, valloader, testloader

#Define the Convolutional Neural Network
class Net(nn.Module):
    def __init__(self, mpk=1, fema=3):
        super().__init__()
        #Images have already gone through a 5*5 MaxPool2d.
        self.pool = nn.MaxPool2d(kernel_size=mpk, stride=mpk)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=fema, kernel_size=(5, 5), stride=1) #1 input chanels, 3 convolution layers, 5 x 5 convolution
        #self.pool2 = nn.MaxPool2d(1, 1)
        #self.conv2 = nn.Conv2d(1, 16, 5)
        #self.avg = nn.AvgPool2d((501, 488))
        width_of_x = calcLengOfConv(length=(calcLengOfConv(length=505, kern_size=mpk, stride_size=mpk)), kern_size=5, stride_size=1)
        width_of_y = calcLengOfConv(length=(calcLengOfConv(length=492, kern_size=mpk, stride_size=mpk)), kern_size=5, stride_size=1)

        self.fc1 = nn.Linear(fema*int(width_of_x*width_of_y), 1)
        #self.fc2 = nn.Linear(120, 84)

    def forward(self, x):
        #x.shape = 505 * 492
        #x = self.pool(x)
        x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #x = self.pool2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        #len(x) = 180048
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x

def make_net(usingCuda, useMultipleGPUs, device):
    # Make the network and have it utilize the gpu
    net = Net()
    # decide what resources the model will use
    if (torch.cuda.device_count() > 1) and useMultipleGPUs:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)
    if usingCuda:
        net.to(device, dtype=torch.float32)
    return net

def descend(trainloader, usingCuda, device, optimizer, net, criterion, train_losses, ftimes):
    train_loss = 0
    t= 0
    for i, data in enumerate(trainloader, 0):
        t = time.time()
        # get the inputs; data is a list of [inputs, labels]
        if usingCuda:
            inputs, labels = data[0].to(device), data[1].to(device)
        else:
            inputs, labels = data

        etime = time.time() - t
        print('took %.4f seconds to separate the tuple "data"' % etime)

        # zero the parameter gradients
        ti = time.time()
        optimizer.zero_grad()
        etime = time.time() - ti
        print('took %.4f seconds to zero the gradients' % etime)

        ti = time.time()
        outputs = net(inputs)  # forward
        etime = time.time() - ti
        print('took %.4f seconds to make a prediction' % etime)

        ti = time.time()
        loss = criterion(outputs, labels)  # compute the loss
        train_loss += loss.item()

        etime = time.time() - ti
        print('took %.4f seconds to compute the loss' % etime)

        ti = time.time()
        loss.backward()  # propogate the error backwards
        etime = time.time() - ti
        print('took %.4f seconds to propagate the loss backwards' % etime)

        ti = time.time()
        optimizer.step()
        etime = time.time() - ti
        print('took %.4f seconds to adjust the parameters' % etime)
    train_losses.append(train_loss)
    ftimes.append(t)

def validate(valloader, device, net, criterion, val_losses):
    val_loss = 0

    # ---
    with torch.no_grad():
        for data in valloader:
            # if usingCuda:
            #     inputs, labels = data[0].to(device), data[1].to(device)
            # else:
            #     inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = net(inputs)
            val_loss += criterion(outputs, labels).item()  # compute the loss
    # ----
    val_losses.append(val_loss)

def train_up_model(useCuda, useMultipleGPUs, device, trainloader, valloader, ftimes, train_losses, val_losses, epochs):
    #Make the network and have it utilize the gpu
    net = make_net(useCuda, useMultipleGPUs, device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.00000000001, momentum=0.3) #lr=0.00000000001, momentum=0.3



    #Train the network
    # training_losses = [] #These lists are necessary for keeping track of the models progress
    # accuracies = []
    # con_accs = []
    # test_losses = []
    # contr_losses = []

    for epoch in range(epochs):  # loop over the dataset multiple times
        descend(trainloader=trainloader, usingCuda=useCuda, device=device, optimizer=optimizer, net=net,
                criterion=criterion, train_losses=train_losses, ftimes=ftimes)

        validate(valloader=valloader, device=device, net=net, criterion=criterion, val_losses=val_losses)

    print('Finished Training')
    return net

def test(testloader, useCuda, device, net, err_margin):
    # Test on the whole dataset
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            if useCuda:
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = data
            outputs = net(inputs)
            print('arrived at predictions')
            total += labels.size(0)
            correct += (abs(outputs - labels) / labels < err_margin).sum().item()
    return correct/total
def save_run(net, train_losses, ftimes, val_losses, accuracy, rootForTrain, rootForTest):
    #Log the statistics of the run and save the model
    pathToModel, log_filename = setupInfoFiles()
    logging.basicConfig(filename=log_filename, encoding='utf-8', level=logging.INFO, filemode='a')
    torch.save(net.state_dict(), pathToModel)

    logging.info(f'Network is: {net}')
    logging.info(f'rootForTrain: {rootForTrain}')
    logging.info(f'rootForTest: {rootForTest}')
    logging.info(f'Epoch finishing times: {ftimes}')
    logging.info(f'Training losses:{train_losses}')
    logging.info(f'Validation losses losses: {val_losses}')
    logging.info(f'Final accuracy: {accuracy}')

def plot_metrics(train_losses, test_losses):
    # show the costs over time
    # x = range(len(training_losses))
    # y = training_losses
    # f, (ax1, ax2) = plt.subplots(2, 1)
    # ax1.plot(x, y)
    #
    # a = range(len(test_losses))
    # ax2.plot(a, test_losses)
    # ax2.plot(a, contr_losses)

    plt.plot(train_losses)
    plt.plot(test_losses)

    plt.show()

def main():
    err_margin = 0.1  # How far off a "correct" prediction can be as a percentage (0.1 is 10%)
    batch_size = 10
    epochs = 20
    useMultipleGPUs = False
    useCuda = True

    rootForTrain = "/mnt/tmpdata/data/isashu/loaderTrainingFiles"
    rootForTest = "/mnt/tmpdata/data/isashu/loaderTestFiles"

    pathToModel = setupInfoFiles()

    # use the gpu
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    trainset, testset = make_datasets(rootForTrain=rootForTrain, rootForTest=rootForTest)
    trainloader, valloader, testloader = make_dataloaders(trainset, testset,batch_size=batch_size)

    ftimes = []
    train_losses = []
    val_losses = []
    net = train_up_model(
        useCuda=useCuda,
        useMultipleGPUs=useMultipleGPUs,
        device=device,
        trainloader=trainloader,
        valloader=valloader,
        ftimes=ftimes,
        train_losses=train_losses,
        val_losses=val_losses,
        epochs=epochs
    )
    accuracy = test(
        testloader=testloader,
        useCuda=useCuda,
        device=device,
        net=net,
        err_margin=err_margin
    )

    save_run(
        net=net,
        train_losses=train_losses,
        ftimes=ftimes,
        val_losses=val_losses,
        accuracy=accuracy,
        rootForTrain=rootForTrain,
        rootForTest=rootForTest
    )
    plot_metrics(train_losses=train_losses, test_losses=val_losses)

    # #Load a model.
    # net = Net()
    # net.load_state_dict(torch.load(pathToModel))

if __name__ == "__main__":
    main()
