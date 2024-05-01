

from PIL import Image
from numpy import load
from numpy import savetxt
import os
import pydicom
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import skimage
from skimage import exposure
import csv
import pandas as pd

data= pd.read_csv('/Users/amelianelson/Desktop/CovidClinical.csv')
ages = data[["to_patient_id","last.status"]]

def preprocess_raw_res(image):
  endImSize = 128
  if image.shape[0] < image.shape[1]:
    newIm = image[1:image.shape[0]:1,1:image.shape[0]:1]
  else:
    newIm = image[1:image.shape[1]:1,1:image.shape[1]:1]
  image = Image.fromarray(newIm)
  res_image = image.resize((endImSize, endImSize))
  finalIm = np.array(res_image)
  #plt.imshow(finalIm)
  l = finalIm.ravel();
  contrast_img = exposure.equalize_hist(finalIm)
  medianOfIm = np.median(contrast_img)
  medianFilteredIm = np.where(contrast_img > medianOfIm, 1.0, 0.0).astype('float32')
  #plt.imshow(medianFilteredIm)
  return medianFilteredIm
                

def preprocess_images(images):
  #images.reshape((images.shape[0], 128, 128, 1))
  #np.where(images > .5, 1.0, 0.0).astype('float32')
  return images.reshape((images.shape[0], 128, 128, 1))
    

trainingArray= []
testingArray= []
with os.scandir("/Users/amelianelson/Desktop/UsefulImages1") as Pictures:
  for picture in Pictures:
    if picture.name != '.DS_Store':
        print(picture.name)
        ds = pydicom.dcmread(picture.path)
        imageDS = ds.pixel_array
        patientStatus = data[data["to_patient_id"].str.contains(picture.name[:7])].iloc[0]["last.status"]
        if patientStatus == 'discharged':
           trainingArray.append(preprocess_raw_res(imageDS))
        else:
           testingArray.append(preprocess_raw_res(imageDS))
TrainingDataArray1 = np.array(trainingArray)
TestingDataArray1 = np.array(testingArray)
del imageDS
del ds
del picture
del TrainingDataArray1
del TestingDataArray1

print('Data 1')

with os.scandir("/Users/amelianelson/Desktop/UsefulImages2") as Pictures:
  for picture in Pictures:
    if picture.name != '.DS_Store':
        print(picture.name)
        ds = pydicom.dcmread(picture.path)
        imageDS = ds.pixel_array
        totalImageArray.append(preprocess_raw_res(imageDS))
DataArray2 = np.array(totalImageArray)
del imageDS
del ds
del picture
del totalImageArray
print('Data 2')

totalImageArray = []
with os.scandir("/Users/amelianelson/Desktop/UsefulImages3") as Pictures:
  for picture in Pictures:
    if picture.name != '.DS_Store':
        print(picture.name)
        ds = pydicom.dcmread(picture.path)
        imageDS = ds.pixel_array
        totalImageArray.append(preprocess_raw_res(imageDS))
DataArray3 = np.array(totalImageArray)
del imageDS
del ds
del picture
del totalImageArray
print('Data 3')

totalImageArray = []
with os.scandir("/Users/amelianelson/Desktop/UsefulImages4") as Pictures:
  for picture in Pictures:
    if picture.name != '.DS_Store':
        print(picture.name)
        ds = pydicom.dcmread(picture.path)
        imageDS = ds.pixel_array
        totalImageArray.append(preprocess_raw_res(imageDS))
DataArray4 = np.array(totalImageArray)
del imageDS
del ds
del picture
del totalImageArray
print('Data 4')

totalDataSet = np.concatenate((DataArray1, DataArray2, DataArray3, DataArray4), axis=0)
#processedSet = preprocess_images(totalDataSet)

trainPercentage = .98
numTrainImages = int(totalDataSet.shape[0] * trainPercentage)
train_images = totalDataSet[1:numTrainImages:1]
test_images = totalDataSet[numTrainImages:totalDataSet.shape[0]:1]
train_processed = preprocess_images(train_images)
test_processed = preprocess_images(test_images)



#np.save('/Users/amelianelson/Desktop/NumpyDataSets/Data4', totalDataArray)
#np.save('/Users/amelianelson/Desktop/NumpyDataSets/Processed4', finished_images)
#tf.data.TFRecordDataset.save(train_dataset,'/Users/amelianelson/Desktop/NumpyDataSets/trainSet')
#tf.data.TFRecordDataset.save(train_dataset,'/Users/amelianelson/Desktop/NumpyDataSets/testSet')


train_size = train_images.shape[0]
train_dataset = (tf.data.Dataset.from_tensor_slices(train_processed).shuffle(train_size).batch(32))
test_size = test_images.shape[0]
test_dataset = (tf.data.Dataset.from_tensor_slices(test_processed).shuffle(test_size).batch(32))


tf.data.TFRecordDataset.save(train_dataset,'/Users/amelianelson/Desktop/NumpyDataSets/trainSet128')
tf.data.TFRecordDataset.save(test_dataset,'/Users/amelianelson/Desktop/NumpyDataSets/testSet128')


print('Completed!')






        