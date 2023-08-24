import pytest
import spotFinder as sf
import setupLoaders
import dataLoader
import condition
import numpy as np
import torch
import os
from IPython import embed

def func(x):
    return x + 1

def make_pro_image():
    img = np.random.randint(7011, size=(2527, 2463)) - 10
    cond_meth = [func for func in filter(callable, condition.__dict__.values())][0]
    pro_img = setupLoaders.processImage(img=img, cond_meth=cond_meth())  # pre-process image
    pro_img = dataLoader.process_img(img=pro_img, useSqrt=True)
    return pro_img

#def save_image(image):

def test_answer():
    assert func(3) == 4

def test_prepro_works():
    #make a numpy array with pilatus dimensions
    img = np.random.randint(7011, size=(2527, 2463)) - 10 #I'm assuming the max pixel value is around 70000 and the min is around -10

    for cond_meth in [func for func in filter(callable, condition.__dict__.values())]:
        print(cond_meth.__name__)
        pro_img = setupLoaders.processImage(img=img, cond_meth=cond_meth())  # pre-process image
        pro_img = dataLoader.process_img(img=pro_img, useSqrt=True)  # live proces
        if pro_img.shape != (1, 832, 832):
            assert False
    assert True
    #preprocess it the same way as setup logger
    #verify it is not just 0

def load_model(path, device):
    arc = sf.Rn(num=18, two_fc_mode=False)
    net = sf.loadModel(path, arc, device=device)
    return net
def test_loadModel(tmp_path):
    device = torch.device(('cuda:' + str(0)) if torch.cuda.is_available() else 'cpu')
    net = load_model("/mnt/tmpdata/data/isashu/thousandYearRun/run_2023-08-13_13_07_40/modelFinal.pth", device=device) #replace with makenet

    #run image through model
    pro_img = make_pro_image().to(device).unsqueeze(dim=0)
    label = net(pro_img)

    #save image, model, and output
    torch.save(pro_img, os.path.join(tmp_path, 'image.pt'))
    torch.save(net.state_dict(), os.path.join(tmp_path, 'model.pth'))

    #loaad image and model
    pro_img = torch.load(os.path.join(tmp_path, 'image.pt'))
    net = load_model(os.path.join(tmp_path, 'model.pth'), device=device)

    new_label = net(pro_img)

    assert new_label == label

def test_consistentLoaders():
    config = {
        "err_margin": 0.1,
        "batch_size": 10,
        "epochs": 500,
        "useMultipleGPUs": False,
        "useCuda": True,
        "lm": False,
        "arc": sf.Rn(),
        "lr": 1e-11,
        "mom": 0.1
    }

    rootForTrain = "/mnt/tmpdata/data/isashu/smallerLoaders/firstSmallerTrainLoader"
    trainloader, valloader = sf.make_trainValLoaders(sf.make_dataset(root=rootForTrain), batch_size=config['batch_size'])

    device = torch.device(('cuda:' + str(0)) if torch.cuda.is_available() else 'cpu')
    net = sf.make_net(device=device, config=config)
    criterion = torch.nn.MSELoss(reduction='mean')
    #optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=config['mom'])  # lr=0.00000000001, momentum=0.3

    sum = 0
    i = 0
    for data in valloader:
        if i == 0:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs)
            print(outputs)
            print(labels)
        i+=1
        loss = criterion(outputs, labels)
    embed()
    i = 0
    for data in valloader:
        if i == 0:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = net(inputs, train=False)
            print(outputs)
            print(labels)
        i+=1

    print(f'Sum of labels {sum}')


