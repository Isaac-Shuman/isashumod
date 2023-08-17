#find . -name training.log > logs
#^Run the above in bash to create the "logs" file

from IPython import embed
import matplotlib.pyplot as plt
def number_extractor(loglines, keyword):
    numbers = [float(l.strip().split()[-1]) for l in loglines if keyword in l]
    return numbers
logs = [l.strip() for l in open('logs', 'r').readlines()]

data = {}
for l in logs:

    lines = open(l, 'r').readlines()
    train_acc = number_extractor(lines, "TRAIN ACCURACIES")
    val_acc = number_extractor(lines, "VAL ACCURACIES")
    uns_acc = number_extractor(lines, "UNSEEN ACCURACIES")

    train_pear = number_extractor(lines, "TRAIN PEARSON")
    val_pear = number_extractor(lines, "VAL PEARSON")
    uns_pear = number_extractor(lines, "UNSEEN PEARSON")

    lr = number_extractor(lines, "lr")
    momentum = number_extractor(lines, "momentum")

    data[l] = train_acc, val_acc, train_pear, val_pear, uns_acc, uns_pear
    a, b, c, d, e, f = data[l]

    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches((6, 3))
    #make it look right with big letters

    axes[0].plot(a, label='train', lw=2)
    axes[0].plot(b, label='val', lw=2)#  uncommennt for top right pics , color="C2")
    axes[0].plot(e, label='uns', lw=2)
    axes[0].set_xlabel("Epochs of training")

    axes[1].plot(c, label='training', lw=2)
    axes[1].plot(d, label='validation', lw=2) #, color="C2")
    axes[1].plot(f, label='uns', lw=2)


    fs = 30
    #axes[0, 0].set_yscale('log')
    axes[0].set_xlabel("Epochs of training", fontsize=fs)
    axes[0].set_ylabel("Accuracies", fontsize=fs)


    axes[1].set_xlabel("Epochs of training", fontsize=fs)
    axes[1].set_ylabel("Pearson correlation coefficients", fontsize=fs)

    for ax in axes:
        ax.tick_params(labelsize=fs)
        #ax.legend(prop={'size': 15}, loc='lower right', fontsize=fs)
        ax.grid(1, ls = '--')

    print('log is %s' % l)
    print('learning rate is %.10f' % lr[0])
    print('momentum is %.10f' % momentum[0])
    plt.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.13, wspace=0.28)
    plt.show()

    '''
    good runs:
    /run_2023-08-04_11_41_21/training.log
    learning rate is 0.000002
    momentum is 0.500000
    
    log is ./run_2023-08-04_14_51_42/training.log
    learning rate is 0.000004
    momentum is 0.000000
    
    log is ./run_2023-08-05_06_07_18/training.log -- best cuz consistent val pearson
    learning rate is 0.000007
    momentum is 0.990000
    
    With maxpool
    log is ./run_2023-08-04_23_22_26/training.log --best overall cuz some validation accuracy
learning rate is   0.000004
momentum is   0.990000



    

    
    '''