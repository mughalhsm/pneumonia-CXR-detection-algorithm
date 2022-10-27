#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import pydicom
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import keras 
from tensorflow import keras 
from skimage.transform import resize
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam


# In[2]:


test1 = pydicom.dcmread('test1.dcm')
test1


# In[3]:


# This function reads in a .dcm file, checks the important fields for our device, and returns a numpy array
# of just the imaging data
def check_dicom(filename): 
    
    print('Load file {} ...'.format(filename))
    ds = pydicom.dcmread(filename)       
    img = ds.pixel_array
    if (ds.Modality != 'DX'or ds.BodyPartExamined != 'CHEST' or (ds.PatientPosition != 'PA' and ds.PatientPosition != 'AP')):
        print('File not compatible')
        return None
    else:
        return img
    
    
# This function takes the numpy array output by check_dicom and 
# runs the appropriate pre-processing needed for our model input
def preprocess_image(img,img_mean,img_std,img_size): 
    image = (img-img_mean)/img_std
    proc_img = resize(image, img_size)
    
    return proc_img

# This function loads in our trained model w/ weights and compiles it 
def load_model(model_path, weight_path):
    json = open(model_path, 'r')
    model_load = json.read()
    json.close()
    model = model_from_json(model_load)
    model.load_weights(weight_path)
    
    loss = 'binary_crossentropy'
    optimizer = Adam(lr = .0001)
    metrics = ['binary_accuracy']
    
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    
    
    return model

# This function uses our device's threshold parameters to predict whether or not
# the image shows the presence of pneumonia using our trained model
def predict_image(model, img, thresh): 
    pred = model.predict(img)
    if pred > thresh:
        print('positive for pneumonia')
    else:
        print('negative for pneumonia')

    
    return pred  


# In[11]:


test_dicoms = ['test1.dcm','test2.dcm','test3.dcm','test4.dcm','test5.dcm','test6.dcm']

model_path = "/home/workspace/my_model.json"
weight_path = "/home/workspace/my_model-02-0.54.hdf5"

IMG_SIZE=(1,224,224,3) # This might be different if you did not use vgg16

my_model = load_model(model_path, weight_path)
thresh = 0.32

# use the .dcm files to test your prediction
for i in test_dicoms:
    dcm = pydicom.dcmread(i)
    img = np.array([])
    img = check_dicom(i)
    print(dcm.Modality)
    print(dcm.BodyPartExamined)
    print(dcm.PatientPosition)
    if img is None:
        continue
    img_mean = np.mean(img)    
    img_std = np.std(img)
    img_proc = preprocess_image(img,img_mean,img_std,IMG_SIZE)
    pred = predict_image(my_model,img_proc,thresh)
    print(pred)
    print( )


# In[7]:


# the last 3 files are not run because the images are not compatible with the algorithm. 
# i.e the images are either not digital radiography (DX), not of the chest or the postion is
# something other than AP or PA view of the xray for example may be lateral of chest. 
# File 4 - Ribcage
# File 5 - CT
# File 6 - Unknown patient postion.


# In[ ]:




