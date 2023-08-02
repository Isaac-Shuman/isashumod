#import statements
from functools import partial
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split
import torchvision
from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler
from dataLoader import CustomImageDataset
import torchvision.transforms.v2 as transforms
from math import floor
from IPython import embed
from torchvision.models import resnet50, vit_b_16,ResNet50_Weights

ERR_MARGIN = 0.1
def calcLengOfConv(length , kern_size, stride_size = -1):
    if stride_size == -1:
        stride_size = kern_size
    return floor((length - kern_size)/stride_size + 1)

#Data loaders
def load_data(trainsets, testsets):
    rootForTrain = "/mnt/tmpdata/data/isashu/smallerLoaders/firstSmallerTrainLoader"
    rootsForTest = ["/mnt/tmpdata/data/isashu/smallLoaders/firstSmallTestLoaders/1.25ALoader",
                    "/mnt/tmpdata/data/isashu/smallLoaders/firstSmallTestLoaders/3.15ALoader",
                    "/mnt/tmpdata/data/isashu/smallLoaders/firstSmallTestLoaders/5.45ALoader"]

    iiFile = "imageNameAndImage.hdf5"
    isFile = "imageNameAndSpots.csv"
    hd_filename = os.path.join(rootForTrain, iiFile)
    cs_filename = os.path.join(rootForTrain, isFile)
    trainset = CustomImageDataset(annotations_file=cs_filename, path_to_hdf5=hd_filename, transform=None)
    trainsets.append(trainset)

    for rootForTest in rootsForTest:
        hd_filename = os.path.join(rootForTest, iiFile)
        cs_filename = os.path.join(rootForTest, isFile)
        testset = CustomImageDataset(annotations_file=cs_filename, path_to_hdf5=hd_filename, transform=None)
        testsets.append(testset)

class Net(nn.Module):
    def __init__(self, mpk=2, mps=2, cvo=5, cvk=5, cvs=1):
        super(Net, self).__init__()
        #self.pool = nn.MaxPool2d(kernel_size=mpk, stride=mps)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=cvo, kernel_size=(cvk, cvk), stride=cvs) #1 input chanels, 5 convolution layers, 5 x 5 convolution
        #self.pool2 = nn.MaxPool2d(1, 1)
        #self.conv2 = nn.Conv2d(1, 16, 5)
        #self.avg = nn.AvgPool2d((501, 488))
        width_of_x = calcLengOfConv(length=1263, kern_size=cvk,
                                    stride_size=cvs)
        width_of_y = calcLengOfConv(length=1231, kern_size=cvk,
                                    stride_size=cvs)

        self.fc1 = nn.Linear(cvo * int(width_of_x * width_of_y), 1)
        #self.fc2 = nn.Linear(120, 84)

    def forward(self, x, train):
        #x = self.pool(x)
        x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        #x = self.pool2(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        return x

class Rn(nn.Module):
    def __init__(self, mpk=1, fema=3):
        super().__init__()

        self.res = resnet50()
        #self.tra = vit_b_16(image_size=832, hidden_dim=1)  #patch_size = 16, and 16 *52 = 832

        self.res.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)#nn.Conv2d(1, self.res.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.fc1 = nn.Linear(1000, 1)
        # self.fc2 = nn.Linear(120, 84)

    def forward(self, x, train):
        # x.shape = 505 * 492
        # x = self.pool(x)
        if train:
            self.res.train()
            #self.tra.train()
        else:
            self.res.eval()
            #self.tra.eval()

        x = self.res(x)
        #x = self.tra(x)
        x = F.relu(x)

        x = F.relu(self.fc1(x))

        return x

class Tr(nn.Module):
    def __init__(self, mpk=1, fema=3):
        super().__init__()

        self.tra = vit_b_16(image_size=832)#, hidden_dim=1)  #patch_size = 16, and 16 *52 = 832
        self.tra.conv_proj = torch.nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))

        self.fc1 = nn.Linear(1000, 1)
        # self.fc2 = nn.Linear(120, 84)

    def forward(self, x, train):
        if train:
            self.tra.train()
        else:
            self.tra.eval()

        x = self.tra(x)
        x = F.relu(x)

        x = F.relu(self.fc1(x))

        return x

#The training function
def train_cifar(config, data_dir=None):
    #net = Net(config["mpk"], config["mps"], config["cvo"], config["cvk"], config["cvs"])
    net = Net(cvo=config["cvo"], cvk=config["cvk"], cvs=config["cvs"]) #EDIT WITH CONFIG FILE
    #have the network utilize the available gpus
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
    net.to(device, dtype = torch.float32) #specifying the dtype here is necessary for some reason

    # Define a Loss function and optimizer
    criterion = nn.MSELoss(reduction='mean')
    #optimizer = optim.SGD(net.parameters(), lr=config["lr"], momentum=config["mo"])
    optimizer = optim.SGD(net.parameters(), lr=4.5e-11, momentum=0.2)


    #I don't know what this block does
    checkpoint = session.get_checkpoint()
    if checkpoint:
        checkpoint_state = checkpoint.to_dict()
        start_epoch = checkpoint_state["epoch"]
        net.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
    else:
        start_epoch = 0

    trainsets=[]
    testsets=[]
    load_data(trainsets=trainsets, testsets=testsets)

    trainset=trainsets[0]

    #Split the training data into training and validation data
    #I haven't looked at it that carefully
    test_abs = int(len(trainset) * 0.5)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    #ask Derek what the "num_workers" parameter is
    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )

    #What is valloader? A: Probably loads the data that was just put into the validation set
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=8
    )

    for epoch in range(start_epoch, 10):  # loop over the dataset multiple times. Is 10 the total number of epochs?
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs, train=True)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = net(inputs, train=False)
                total += labels.size(0)
                correct += (abs(outputs - labels)/labels < ERR_MARGIN).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        #Communicate with Ray Tune. Again, I don't really know how the checkpoints work
        checkpoint_data = {
            "epoch": epoch,
            "net_state_dict": net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"loss": val_loss / val_steps, "accuracy": correct / total},
            checkpoint=checkpoint,
        )

    print("Finished Training")

#test set accuracy
def test_accuracy(net, device="cpu", testset=[]):
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs, train=False)
            total += labels.size(0)
            correct += (abs(outputs - labels) / labels < ERR_MARGIN).sum().item()
    return correct / total

def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    data_dir = os.path.abspath("./data")

    #What hyperparameter values would we like to try
    #Every time you edit this file you must edit 2 other locations
    #1. 1st line under "def train_cifar(config, data_dir=None)"
    #2. best_trained_model = Net(...
    config = {
        "mpk": tune.choice([2*i for i in range(1,10)]),
        "mps": tune.choice([2*i for i in range(1,10)]),
        "cvo": tune.choice([2**i for i in range(9)]),
        "cvk": tune.choice([2**i for i in range(9)]),

        "cvs": tune.choice([2**i for i in range(9)]),

        "lr": tune.loguniform(1e-13, 1e-1), #By default has  a range of 10^3
        "batch_size": tune.choice([10]),
        "mo": tune.choice([0.1*i for i in range(10)])
    }

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2,
    ) #stops trials early
    result = tune.run(
        partial(train_cifar, data_dir=data_dir),
        resources_per_trial={"cpu": 48, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
    )

    embed()
    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")

    best_trained_model = Net(cvo=best_trial.config["cvo"], cvk=best_trial.config["cvk"], cvs=best_trial.config["cvs"])#config["mpk"], config["mps"], config["cvo"], config["cvk"], config["cvs"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    best_checkpoint_data = best_checkpoint.to_dict()

    best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    trainsets = []
    testsets = []
    load_data(trainsets=trainsets, testsets=testsets)
    for testset in testsets:
        test_acc = test_accuracy(net=best_trained_model, device=device, testset=testset)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=2, max_num_epochs=20, gpus_per_trial=1)  # num_samples = # of trials