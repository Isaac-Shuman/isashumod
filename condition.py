

import torch

def tester():
    print("he")

def mx_alpha():
    return torch.nn.MaxPool2d(5, stride=1)


def mx_beta():
    return torch.nn.MaxPool2d(3, stride=1)

def nothing():
    return torch.nn.Identity()

def resize_alpha():
    '''
    Forces the dimensions to half the size of the images from
    '''
    return torch.transforms.Resize(1263, 1231)





