

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
#file creates a batch for training that is completely made up of discharged patients and two testing batches,
#one of discharged patients and the other if deceased patients.

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
trainingArray= []
testingArray= []
with os.scandir("/Users/amelianelson/Desktop/UsefulImages1") as Pictures:
  for picture in Pictures:
    if picture.name != '.DS_Store':
        print(picture.name)
        ds = pydicom.dcmread(picture.path)
        imageDS = ds.pixel_array
        #Finds patients outcome in order to discern whether part of training or testing set
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
del trainingArray
del testingArray

#Preprocessing second fourth of data from DICOM to numpy image array
trainingArray= []
testingArray= []
with os.scandir("/Users/amelianelson/Desktop/UsefulImages2") as Pictures:
  for picture in Pictures:
    if picture.name != '.DS_Store':
        print(picture.name)
        ds = pydicom.dcmread(picture.path)
        imageDS = ds.pixel_array
        #Finds patients outcome in order to discern whether part of training or testing set
        patientStatus = data[data["to_patient_id"].str.contains(picture.name[:7])].iloc[0]["last.status"]
        if patientStatus == 'discharged':
           trainingArray.append(preprocess_raw_res(imageDS))
        else:
           testingArray.append(preprocess_raw_res(imageDS))
TrainingDataArray2 = np.array(trainingArray)
TestingDataArray2 = np.array(testingArray)
del imageDS
del ds
del picture
del trainingArray
del testingArray

#Preprocessing third fourth of data from DICOM to numpy image array
trainingArray= []
testingArray= []
with os.scandir("/Users/amelianelson/Desktop/UsefulImages3") as Pictures:
  for picture in Pictures:
    if picture.name != '.DS_Store':
        print(picture.name)
        ds = pydicom.dcmread(picture.path)
        imageDS = ds.pixel_array
        #Finds patients outcome in order to discern whether part of training or testing set
        patientStatus = data[data["to_patient_id"].str.contains(picture.name[:7])].iloc[0]["last.status"]
        if patientStatus == 'discharged':
           trainingArray.append(preprocess_raw_res(imageDS))
        else:
           testingArray.append(preprocess_raw_res(imageDS))
TrainingDataArray3 = np.array(trainingArray)
TestingDataArray3 = np.array(testingArray)
del imageDS
del ds
del picture
del trainingArray
del testingArray

#Preprocessing fourth fourth of data from DICOM to numpy image array
trainingArray= []
testingArray= []
with os.scandir("/Users/amelianelson/Desktop/UsefulImages4") as Pictures:
  for picture in Pictures:
    if picture.name != '.DS_Store':
        print(picture.name)
        ds = pydicom.dcmread(picture.path)
        imageDS = ds.pixel_array
        #Finds patients outcome in order to discern whether part of training or testing set
        patientStatus = data[data["to_patient_id"].str.contains(picture.name[:7])].iloc[0]["last.status"]
        if patientStatus == 'discharged':
           trainingArray.append(preprocess_raw_res(imageDS))
        else:
           testingArray.append(preprocess_raw_res(imageDS))
TrainingDataArray4 = np.array(trainingArray)
TestingDataArray4 = np.array(testingArray)
del imageDS
del ds
del picture
del trainingArray
del testingArray

#Combining Entire data sets by patient outcome
train_images = np.concatenate((TrainingDataArray1,TrainingDataArray2, TrainingDataArray3, TrainingDataArray4), axis=0)
test_images = np.concatenate((TestingDataArray1,TestingDataArray2, TestingDataArray3, TestingDataArray4), axis=0)

#seperating out training and testing numpy arrays
train_processed = preprocess_images(train_images[1:int(train_images.shape[0]*.90):1])
test_deseased_processed = preprocess_images(test_images)
test_discharged_processed = preprocess_images(train_images[int(train_images.shape[0]*.90):train_images.shape[0]:1])

#Creaing training and testing data sets
train_size = train_processed.shape[0]
train_dataset = (tf.data.Dataset.from_tensor_slices(train_processed).shuffle(train_size).batch(32))
test_deseased_size = test_deseased_processed.shape[0]
test_deseased_dataset = (tf.data.Dataset.from_tensor_slices(test_deseased_processed).shuffle(test_deseased_size).batch(32))
test_discharged_size = test_discharged_processed.shape[0]
test_discharged_dataset = (tf.data.Dataset.from_tensor_slices(test_discharged_processed).shuffle(test_discharged_size).batch(32))


#Saving Data sets
tf.data.TFRecordDataset.save(train_dataset,'/Users/amelianelson/Desktop/NumpyDataSets/trainSet128Healthy')
tf.data.TFRecordDataset.save(test_deseased_dataset,'/Users/amelianelson/Desktop/NumpyDataSets/testSetDeceased128')
tf.data.TFRecordDataset.save(test_discharged_dataset,'/Users/amelianelson/Desktop/NumpyDataSets/testSetDischarged128')

#Reporting size of data sets
print(str(train_size))
print(str(test_deseased_size))
print(str(test_discharged_size))


print('Completed!')






        