
from PIL import Image
from numpy import load
from numpy import savetxt
import os
import pydicom
import numpy as np
                
def preprocess_raw_res(image):
  endImSize = 512
  if image.shape[0] < image.shape[1]:
    newIm = image[1:image.shape[0]:1,1:image.shape[0]:1]
  else:
    newIm = image[1:image.shape[1]:1,1:image.shape[1]:1]
  image = Image.fromarray(newIm)
  res_image = image.resize((endImSize, endImSize))
  finalIm = np.array(res_image)
  return finalIm

def preprocess_images(images):
  images = images.reshape((images.shape[0], 512, 512, 1)) / 255.
  return np.where(images > .5, 1.0, 0.0).astype('float32')
    

totalImageArray = []
with os.scandir("/Users/amelianelson/Desktop/UsefulImages5") as Pictures:
  for picture in Pictures:
    if picture.name != '.DS_Store':
        print(picture.name)
        ds = pydicom.dcmread(picture.path)
        imageDS = ds.pixel_array
        totalImageArray.append(preprocess_raw_res(imageDS))
totalDataArray = np.array(totalImageArray)
del imageDS
del ds
del picture
print('Completedfirst')

np.save('/Users/amelianelson/Desktop/NumpyDataSets/Data5', totalDataArray)

train_images = preprocess_images(totalDataArray)
print('Completed!')






        