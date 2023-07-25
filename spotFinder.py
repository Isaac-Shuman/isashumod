ERR_MARGIN = 0.1 #How far off a "correct" prediction can be as a percentage (0.1 is 10%)
batch_size = 10
EPOCHS =20
useMultipleGPUs = False
usingCuda = True

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

#Setup for the run info files
def setupInfoFiles():
    rootFolderName = '/mnt/tmpdata/data/isashu/runFolders'
    folderName = os.path.join(rootFolderName,str(time.strftime('run_%Y-%m-%d_%H:%M:%S')))
    logName = 'metrics.log'

    os.mkdir(folderName)
    log_filename = os.path.join(folderName, logName)
    logging.basicConfig(filename=log_filename, encoding='utf-8', level=logging.INFO, filemode='a')

    pathToModel = os.path.join(folderName, 'model.pth')

    return pathToModel


pathToModel = setupInfoFiles()

#use the gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



#Used to calculate the dimensions of an image after it passed through a pool or convolution layer
def calcLengOfConv(length , kern_size, stride_size = -1):
    if stride_size == -1:
        stride_size = kern_size
    return floor((length - kern_size)/stride_size + 1)

#Make the data loaders
def make_datasets():
    rootForTrain = "/mnt/tmpdata/data/isashu/loaderTrainingFiles"
    rootForTest = "/mnt/tmpdata/data/isashu/loaderTestFiles"
    iiFile = "imageNameAndImageDS.hdf5"
    isFile = "imageNameAndSpots.csv"
    hd_filename = os.path.join(rootForTrain, iiFile)
    cs_filename = os.path.join(rootForTrain, isFile)
    trainset = CustomImageDataset(annotations_file=cs_filename, path_to_hdf5=hd_filename, transform=None)

    hd_filename = os.path.join(rootForTest, iiFile)
    cs_filename = os.path.join(rootForTest, isFile)
    testset = CustomImageDataset(annotations_file=cs_filename, path_to_hdf5=hd_filename, transform=None)

    return trainset, testset

trainset, testset = make_datasets()
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False)


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


#Make the network and have it utilize the gpu
net = Net()
logging.info(f'Network is: {net}')
torch.save(net.state_dict(), pathToModel) #save the model

if (torch.cuda.device_count()> 1) and useMultipleGPUs:
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    net = nn.DataParallel(net)
if usingCuda:
    net.to(device, dtype = torch.float32)

#Define a Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.00000000001, momentum=0.3) #lr=0.00000000001, momentum=0.3

#Train the network
training_losses = [] #These lists are necessary for keeping track of the models progress
accuracies = []
con_accs = []
test_losses = []
contr_losses = []
ftimes = []
for epoch in range(EPOCHS):  # loop over the dataset multiple times
    training_loss = 0.0
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
        outputs = net(inputs) #forward
        etime = time.time() - ti
        print('took %.4f seconds to make a prediction' % etime)

        ti = time.time()
        loss = criterion(outputs, labels) #compute the loss
        training_loss += loss.item()


        etime = time.time() - ti
        print('took %.4f seconds to compute the loss' % etime)

        ti = time.time()
        loss.backward() #propogate the error backwards
        etime = time.time() - ti
        print('took %.4f seconds to propagate the loss backwards' % etime)

        ti = time.time()
        optimizer.step()
        etime = time.time() - ti
        print('took %.4f seconds to adjust the parameters' % etime)
    training_losses.append(training_loss)
    ftimes.append(time.time())

    total = 0
    correct = 0
    cont_cor = 0
    test_loss = 0
    contr_loss =0
    with torch.no_grad():
        for data in testloader:
            if usingCuda:
                inputs, labels = data[0].to(device), data[1].to(device)
            else:
                inputs, labels = data
            # calculate outputs by running images through the network
            outputs = net(inputs)
            test_loss += criterion(outputs, labels).item()  # compute the loss
            print('arrived at predictions')
            total += labels.size(0)
            correct += (abs(outputs - labels)/labels < ERR_MARGIN).sum().item()

            #Check how guessing the mean value of spots compares. Mean is 402.8 spots
            control = torch.zeros_like(outputs)
            control.add_(403.)
            contr_loss += criterion(control, labels).item()
            cont_cor += (abs(control - labels)/labels < ERR_MARGIN).sum().item()

    accuracies.append(100 * correct // total)
    con_accs.append(100 * cont_cor//total)

    test_losses.append(test_loss)
    contr_losses.append(contr_loss)

print('Finished Training')

# #Load a model.
# net = Net()
# net.load_state_dict(torch.load(pathToModel))

#Test on the whole dataset
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        if usingCuda:
            inputs, labels = data[0].to(device), data[1].to(device)
        else:
            inputs, labels = data
        # calculate outputs by running images through the network
        outputs = net(inputs)
        # the class with the highest energy is what we choose as prediction
        #_, predicted = torch.max(outputs.data, 1)
        print('arrived at predictions')
        total += labels.size(0)
        correct += (abs(outputs - labels)/labels < ERR_MARGIN).sum().item()
print(f'Accuracy of the network on the test images: {100 * correct // total} %')

#log the run
logging.info(f'Epoch finishing times: {ftimes}')
logging.info(f'Training losses:{training_losses}')
logging.info(f'Test losses: {test_losses}')

#show the costs over time
x = range(len(training_losses))
y = training_losses
f, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(x, y)

a = range(len(test_losses))
ax2.plot(a, test_losses)
ax2.plot(a, contr_losses)

plt.show()
