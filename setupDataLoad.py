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

'''
This script creates:
1. An hdf5 file that stores the images(as matrices) for each .exp file
2. A csv file that stores the number of spots for each .exp file
'''
# 1. Change the directory name
raw_dir = '/mnt/tmpdata/data/test_spots/for_isaac'
pro_dir = '/mnt/tmpdata/data/isashu/fourthLoader/trainFiles'
# Open an hdf file
hd_filename = os.path.join(pro_dir, "imageNameAndImage.hdf5")
cs_filename = os.path.join(pro_dir, "imageNameAndSpots.csv")
hd = h5py.File(hd_filename, "w")

cond_meth_name = "resize_alpha"
cond_meth = getattr(condition, cond_meth_name)()

cs = open(cs_filename, 'w')
writer = csv.writer(cs)

num_files = len(os.listdir(raw_dir))
per_files = 0.8
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

        cond_img = torch.tensor(raw_img.astype(numpy.float32)).unsqueeze(0).unsqueeze(0)
        cond_img = cond_meth(cond_img)#.squeeze()
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

