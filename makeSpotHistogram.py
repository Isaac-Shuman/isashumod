dir2 = '/mnt/tmpdata/data/isashu/actualLoaderFiles'
hd_filename = os.path.join(dir2, "imageNameAndImage.hdf5") #Complete path and name of hdf5 file
cs_filename = os.path.join(dir2, "imageNameAndSpots.csv") #Complete path and name of csv file

data = CustomImageDataset(cs_filename, hd_filename)

pixelValues = []
for i in range(1):
    da = data[i][0]
    print(da)
    pixelValues += list(da.flatten())

        #spotCounts.append((data.img_labels.iloc[i, 0], data.img_labels.iloc[i, 1]))
    #spotCounts.append(data[i][1])
#print(spotCounts)
plt.hist(pixelValues, 100000)
plt.show()