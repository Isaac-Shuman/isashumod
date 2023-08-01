import torch
from torch.utils.data import Dataset
from torchvision import datasets
import os
import pandas as pd
import h5py
from matplotlib import pyplot as plt
import numpy
from IPython import embed
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights #Added for ResNet
import torchvision
'''
annotations_file should be the full path of the csv file with rows [image name, # of spots]
'''

#LAL = 1 #Use 1/LAL of the dataset
class CustomImageDataset(Dataset):

    def __init__(self, annotations_file, path_to_hdf5, transform=None, target_transform=None, test=False):
        self.img_labels = pd.read_csv(annotations_file)
        self.path_to_hdf5 = path_to_hdf5
        self.transform = transform
        self.target_transform = target_transform
        self.test = test

    def __len__(self):
        return len(self.img_labels)#//LAL

    def __getitem__(self, idx):
        #get the image as a numpy array
        #The first 300 items are reserved for the testing data set
        idx_offset = 0
        #if not self.test:
         #   idx_offset += 300
        with h5py.File(self.path_to_hdf5, 'r') as f:
            image = f[self.img_labels.iloc[idx + idx_offset, 0]][()]
            #image = torch.zeros(size=(1, 1263, 1231), dtype=torch.float32) #Used for transform to test having 832*832
        #get the number of spots
        label = self.img_labels.iloc[idx + idx_offset, 1]
        #label = torch.from_numpy(label)
        label = torch.tensor([label]).type(torch.float32) #label transformation

        #stuff
        if self.transform:
            lt = 0
            # up = 20000  # the upper threshold is actually this value + 60
            # image[image < lt] = lt
            # image = torch.tensor(image)
            # #image[image > up] = up

            #image = image.squeeze(dim=0)
            # if len(image.shape) == 2:
            #     image = image[None]  # this is called broadcasting, adds an extra dimension
            #image = torch.tensor(image.astype(numpy.float32)) #converts the 3d numpy integer array into a 3d floating point tensor

            #image = self.transform(image).unsqueeze(dim=0) #Added for ResNet
            #image = numpy.repeat(image, 3, axis=0) #take this out and modify the ResNet
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

'''
# from IPython import embed;
# embed()

dir2 = '/mnt/tmpdata/data/isashu/actualLoaderFiles'
hd_filename = os.path.join(dir2, "imageNameAndImage.hdf5") #Complete path and name of hdf5 file
cs_filename = os.path.join(dir2, "imageNameAndSpots.csv") #Complete path and name of csv file

data = CustomImageDataset(cs_filename, hd_filename)

pixelValues = []
for i in range(1):
    da = data[i][0]
    print(da)
    pixelValues += list(da.flatten())

        #spotCounts.append((data.img_labels.iloc[i, 0], data.img_labels.iloc[i, 1]))
    #spotCounts.append(data[i][1])
#print(spotCounts)
plt.hist(pixelValues, 100000)
plt.show()

#
# #print(data[0][1])
'''
