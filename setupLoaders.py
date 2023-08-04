import os
import csv
import h5py
from dxtbx.model import ExperimentList
from dials.array_family import flex
import condition
from IPython import embed
import numpy
import torch
import time

def processImage(img, cond_meth):
    '''
    img is a numpy array
    returns a tensor
    '''
    cond_img = torch.tensor(img.astype(numpy.float32)).unsqueeze(0)
    cond_img = cond_meth(cond_img)  # .squeeze()

    lt = 0
    cond_img[cond_img < lt] = lt

    return cond_img

def setupTrainLoader(raw_big_dir, raw_directories, pro_big_dir, per_files, cond_meth_name):

    # make everything go in the for loop and delete the current pro_dir
    ini = time.time()

    pro_dir = pro_big_dir
    # Open an hdf file
    hd_filename = os.path.join(pro_dir, "imageNameAndImage.hdf5")
    cs_filename = os.path.join(pro_dir, "imageNameAndSpots.csv")

    hd = h5py.File(hd_filename, "w-")  # The w- should cause this command to fail if the file already exists

    cond_meth = getattr(condition, cond_meth_name)()

    cs = open(cs_filename, 'w')
    writer = csv.writer(cs)

    for di in raw_directories:  # for direct in raw_directiories
        print(di)
        # generate the path...
        raw_dir = os.path.join(raw_big_dir, di)  # 1.for the raw directory
        num_files = len(os.listdir(raw_dir))
        firstFiles = True  # True if you are making the dataset from the first several files as opposed to the last several files

        i = 0
        for filename in os.listdir(raw_dir):
            i += 1
            print(filename)
            #This split may be imperfect
            # If we want the first per_files*100 percent of files, break the for loop as soon as i is past a certain index
            if firstFiles and i > per_files * num_files:
                break
            # If we want the last per_files*100 percent of files, don't load the files until i is past a certain index
            if not firstFiles and i <= (1 - per_files) * num_files + 1:
                continue

            if filename.endswith('.expt'):
                # 4.Extract the numpy image from the experiment file
                # What is the experiment list?
                El = ExperimentList.from_file(os.path.join(raw_dir, filename))
                raw_img = El[0].imageset.get_raw_data(0)[0].as_numpy_array()

                cond_img = processImage(raw_img, cond_meth)
                # 5.Store the numpy image in the hdf file
                dset = hd.create_dataset(filename, data=cond_img)

                # Extract the # of spots from the corresponding refl files
                refl_fname = filename.replace(".expt", ".refl")
                R = flex.reflection_table.from_file(os.path.join(raw_dir, refl_fname))
                num_spot = len(R)

                # write a row to the csv file
                writer.writerow([filename, num_spot])
        print(f'Time is: {time.time() - ini}')

    cs.close()
    hd.attrs["condition_method_name"] = cond_meth_name
    hd.attrs["root folder"] = raw_big_dir
    hd.close()


def setupTestLoaders(raw_big_dir, raw_directories, pro_big_dir, per_files, cond_meth_name):
    cond_meth = getattr(condition, cond_meth_name)()

    for di in raw_directories: #for direct in raw_directiories
    #generate the path...
        raw_dir = os.path.join(raw_big_dir, di)#1.for the raw directory
        pro_dir = os.path.join(pro_big_dir, di[:5] + 'Loader')#2.for the processed directory
        os.mkdir(pro_dir)

        # Open an hdf file
        hd_filename = os.path.join(pro_dir, "imageNameAndImage.hdf5")
        cs_filename = os.path.join(pro_dir, "imageNameAndSpots.csv")
        hd = h5py.File(hd_filename, "w-")

        cs = open(cs_filename, 'w')
        writer = csv.writer(cs)

        num_files = len(os.listdir(raw_dir))
        firstFiles = True #True if you are making the dataset from the first several files as opposed to the last several files

        i = 0
        for filename in os.listdir(raw_dir):
            i += 1
            print(filename)
            # If we want the first per_files*100 percent of files, break the for loop as soon as i is past a certain index
            if firstFiles and i > per_files * num_files:
                break
            # If we want the last per_files*100 percent of files, don't load the files until i is past a certain index
            if not firstFiles and i <= (1 - per_files)*num_files + 1:
                continue

            if filename.endswith('.expt'):
                # 4.Extract the numpy image from the experiment file
                # What is the experiment list?
                El = ExperimentList.from_file(os.path.join(raw_dir, filename))
                raw_img = El[0].imageset.get_raw_data(0)[0].as_numpy_array()

                cond_img = processImage(img=raw_img, cond_meth=cond_meth)
                # 5.Store the numpy image in the hdf file
                dset = hd.create_dataset(filename, data=cond_img)

                # Extract the # of spots from the corresponding refl files
                refl_fname = filename.replace(".expt", ".refl")
                R = flex.reflection_table.from_file(os.path.join(raw_dir, refl_fname))
                num_spot = len(R)

                # write a row to the csv file
                writer.writerow([filename, num_spot])

        cs.close()
        hd.attrs["condition_method_name"] = cond_meth_name
        hd.attrs["root folder"] = raw_dir
        hd.close()


def main():
    new_loaders = 'threeDown'
    all_loaders = '/mnt/tmpdata/data/isashu/newLoaders'
    big_dir = os.path.join(all_loaders, new_loaders)
    os.mkdir(big_dir)

    loaderSizes =['smaller','small','big']
    loaderSizeVals = [0.01, 0.1, 1]

    raw_big_dir = '/mnt/tmpdata/data/isashu/exptFileDumps'
    raw_train_directories = ['1.42ADump', '1.450ADump', '1.45ADump', '1.50ADump', '1.60ADump', '1.62ADump',
                       '1.66ADump', '1.70ADump', '1.72ADump', '1.74ADump', '1.76ADump', '1.81ADump',
                       '1.85ADump', '1.8ADump', '1.95ADump', '2.0ADump', '2.5ADump', '2.80ADump', '2.85ADump',
                       '2.90ADump', '5.40ADump']
    raw_test_directories = ['1.25ADump', '3.15ADump', '5.45ADump']

    cond_meth_name = "resize_for_trans"

    i = 0
    for size in loaderSizes:
        loaders = os.path.join(big_dir, size + 'Loaders')
        os.mkdir(loaders)

        train_loader = os.path.join(loaders, 'trainLoader')
        os.mkdir(train_loader)
        setupTrainLoader(raw_big_dir=raw_big_dir,raw_directories=raw_train_directories, pro_big_dir=train_loader,
                         per_files=loaderSizeVals[i], cond_meth_name=cond_meth_name)

        test_loader = os.path.join(loaders, 'testLoaders')
        os.mkdir(test_loader)
        setupTestLoaders(raw_big_dir, raw_test_directories, pro_big_dir=test_loader, per_files=loaderSizeVals[i],
                         cond_meth_name=cond_meth_name )

        val_loader = os.path.join(loaders, 'valLoader')
        os.mkdir(val_loader)
        setupTrainLoader(raw_big_dir=raw_big_dir, raw_directories=raw_test_directories, pro_big_dir=val_loader,
                         per_files=loaderSizeVals[i], cond_meth_name=cond_meth_name)

        i += 1

if __name__ == "__main__":
    main()

