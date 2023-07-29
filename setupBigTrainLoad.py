#export HDF5_USE_FILE_LOCKING=FALSE

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

'''
This script creates:
1. An hdf5 file that stores the images(as matrices) for each .exp file
2. A csv file that stores the number of spots for each .exp file
'''
# 1. Change the directory
raw_big_dir = '/mnt/tmpdata/data/isashu/exptFileDumps'
raw_directories = ['1.42ADump', '1.450ADump', '1.45ADump', '1.50ADump', '1.60ADump', '1.62ADump',
                   '1.66ADump', '1.70ADump', '1.72ADump', '1.74ADump', '1.76ADump', '1.81ADump',
                   '1.85ADump', '1.8ADump', '1.95ADump', '2.0ADump', '2.5ADump', '2.80ADump', '2.85ADump', '2.90ADump', '5.40ADump']

pro_big_dir = '/mnt/tmpdata/data/isashu/smallerLoaders/firstSmallerTrainLoader'

#make everything go in the for loop and delete the current pro_dir
ini = time.time()


pro_dir = pro_big_dir
# Open an hdf file
hd_filename = os.path.join(pro_dir, "imageNameAndImage.hdf5")
cs_filename = os.path.join(pro_dir, "imageNameAndSpots.csv")


hd = h5py.File(hd_filename, "w-") #The w- should cause this command to fail if the file already exists

cond_meth_name = "resize_alpha"
cond_meth = getattr(condition, cond_meth_name)()

cs = open(cs_filename, 'w')
writer = csv.writer(cs)

for di in raw_directories: #for direct in raw_directiories
    print(di)
    # generate the path...
    raw_dir = os.path.join(raw_big_dir, di)  # 1.for the raw directory
    num_files = len(os.listdir(raw_dir))
    per_files = 0.01
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

            cond_img = torch.tensor(raw_img.astype(numpy.float32)).unsqueeze(0)
            cond_img = cond_meth(cond_img)#.squeeze()
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
