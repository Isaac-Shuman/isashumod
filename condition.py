

import torch


def mx_alpha():
    return torch.nn.MaxPool2d(5, stride=1)


def mx_beta():
    return torch.nn.MaxPool2d(3, stride=1)





