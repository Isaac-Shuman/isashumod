#export HDF5_USE_FILE_LOCKING=FALSE

#TODO
#1.Add the ability to specify what percentage of the files you would like to make the dataset for
#2.Test it

import os
import csv
import h5py
from dxtbx.model import ExperimentList
from dials.array_family import flex
import condition
from IPython import embed



exit()
'''
h.attrs["condition_method_name"] = cond_meth_name
    for i_img, raw_img in enumerate(raw_imgs): # loop over expts or whatever
        cond_img = cond_meth(raw_img)
        h.create_dataset("something_%d.expt" % i_img, data=cond_img) # e.g.
'''

'''
This script creates:
1. An hdf5 file that stores the images(as matrices) for each .exp file
2. A csv file that stores the number of spots for each .exp file
'''
# 1. Change the directory name
raw_dir = '/mnt/tmpdata/data/test_spots/for_isaac'
pro_dir = '/mnt/tmpdata/data/isashu/secLoaderTrainingFiles'
# Open an hdf file
hd_filename = os.path.join(pro_dir, "imageNameAndImage.hdf5")
cs_filename = os.path.join(pro_dir, "imageNameAndSpots.csv")
hd = h5py.File(hd_filename, "w")

cond_meth_name = "nothing"
cond_meth = getattr(condition, cond_meth_name)()
hd.attrs["condition_method_name"] = cond_meth_name

cs = open(cs_filename, 'w')
writer = csv.writer(cs)
bo = True
for filename in os.listdir(raw_dir):
    print(filename)
    if bo:
        if filename == "idx-Try4mgA1n_1_00299_strong.expt":
            bo = False
        continue

    print(filename + " was accepted")

    if filename.endswith('.expt'):
        # 4.Extract the numpy image from the experiment file
        # What is the experiment list?
        El = ExperimentList.from_file(os.path.join(raw_dir, filename))
        raw_img = El[0].imageset.get_raw_data(0)[0].as_numpy_array()

        cond_img = cond_meth(raw_img)
        # 5.Store the numpy image in the hdf file
        dset = hd.create_dataset(filename, data=cond_img)

        # Extract the # of spots from the corresponding refl files
        refl_fname = filename.replace(".expt", ".refl")
        R = flex.reflection_table.from_file(os.path.join(raw_dir, refl_fname))
        num_spot = len(R)

        # write a row to the csv file
        writer.writerow([filename, num_spot])

cs.close()
hd.close()
# 6. Close the hdf file
