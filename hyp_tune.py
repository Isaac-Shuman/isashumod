from isashumod import spotFinder
import random
import numpy as np
import sys
import time


outdir = sys.argv[1]
num_ep = int(sys.argv[2])
loaderRoot = "/mnt/tmpdata/data/isashu/newLoaders/threeDown/smallLoaders/"

def gen_params():
    num = random.choice([18, 34, 50])
    optim = 0
    lr = np.exp(np.random.uniform(np.log(1e-7), np.log(1e-3)))
    mom = random.choice([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
    two_fc_mode = random.choice([0,1])

    params = {
        'num': num,
        'optim': optim,
        'lr': lr,
        'mom': mom,
        'two_fc_mode': bool(two_fc_mode)
    }

    return params


while True:
    params = gen_params()
    print(params)
    try:
        spotFinder.main(outdir,loaderRoot,epochs=num_ep, **params) 
    except Exception as err:
        print(str(err))
        pass
    print("waiting 10 sec for GPU to clear... ") 
    time.sleep(10)
