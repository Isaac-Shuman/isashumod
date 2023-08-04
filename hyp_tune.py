from isashumod import spotFinder
import random
import numpy as np

def gen_params():
    num = random.choice([18, 34, 50])
    optim = 0
    lr = np.exp(np.random.uniform(np.log(1e-7), np.log(1e-3)))
    mom = random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])

    params = {
        'num': num,
        'optim': optim,
        'lr': lr,
        'mom': mom
    }

    return params

while True:
    params = gen_params()
    spotFinder.main(**params) #num=18, lr=1e-7, optim=0, mom=0.99