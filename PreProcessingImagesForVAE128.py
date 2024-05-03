

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

#This file takes in the DICOM images created in the PreProcessingFromNBIADataRetriver.py and creates
#testing and training sets. For the experiment the data set was split into four parts for handling. This
#file creates data set that are 98% training regardless of patient status for testing.

# Reading in the Covid Clinical CSV file that is associated with the COVID-19-NY-SBU describing patinet status.
data= pd.read_csv('/Users/amelianelson/Desktop/CovidClinical.csv')
ages = data[["to_patient_id","last.status"]]

def preprocess_raw_res(image):
  endImSize = 128
  #Making image into a square of smallest side (top or right side)
  if image.shape[0] < image.shape[1]:
    newIm = image[1:image.shape[0]:1,1:image.shape[0]:1]
  else:
    newIm = image[1:image.shape[1]:1,1:image.shape[1]:1]
  image = Image.fromarray(newIm)
  #Resizing image to be 128 by 128
  res_image = image.resize((endImSize, endImSize))
  finalIm = np.array(res_image)
  #Creating contrast within image
  l = finalIm.ravel();
  contrast_img = exposure.equalize_hist(finalIm)
  medianOfIm = np.median(contrast_img)
  #Binarizing image by making those above median pixel value 1 and the rest 0
  medianFilteredIm = np.where(contrast_img > medianOfIm, 1.0, 0.0).astype('float32')
  return medianFilteredIm

#Reshaping data set to meet shape requirements for running VAE                
def preprocess_images(images):
  return images.reshape((images.shape[0], 128, 128, 1))
    
#Preprocessing first fourth of data from DICOM to numpy image array
totalImageArray = []
with os.scandir("/Users/amelianelson/Desktop/UsefulImages1") as Pictures:
  for picture in Pictures:
    if picture.name != '.DS_Store':
        print(picture.name)
        ds = pydicom.dcmread(picture.path)
        imageDS = ds.pixel_array
        totalImageArray.append(preprocess_raw_res(imageDS))
DataArray1 = np.array(totalImageArray)
del imageDS
del ds
del picture
del totalImageArray
print('Data 1')

#Preprocessing second fourth of data from DICOM to numpy image array
totalImageArray = []
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

#Preprocessing third fourth of data from DICOM to numpy image array
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

#Preprocessing fourth fourth of data from DICOM to numpy image array
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

#Combining Entire Data Set
totalDataSet = np.concatenate((DataArray1, DataArray2, DataArray3, DataArray4), axis=0)


#Setting aside 98% of data set for training 
trainPercentage = .98
numTrainImages = int(totalDataSet.shape[0] * trainPercentage)
train_images = totalDataSet[1:numTrainImages:1]
test_images = totalDataSet[numTrainImages:totalDataSet.shape[0]:1]
train_processed = preprocess_images(train_images)
test_processed = preprocess_images(test_images)

#Creating test and train batches
train_size = train_images.shape[0]
train_dataset = (tf.data.Dataset.from_tensor_slices(train_processed).shuffle(train_size).batch(32))
test_size = test_images.shape[0]
test_dataset = (tf.data.Dataset.from_tensor_slices(test_processed).shuffle(test_size).batch(32))


#Saving test and train batches
tf.data.TFRecordDataset.save(train_dataset,'/Users/amelianelson/Desktop/NumpyDataSets/trainSet128')
tf.data.TFRecordDataset.save(test_dataset,'/Users/amelianelson/Desktop/NumpyDataSets/testSet128')


print('Completed!')






        