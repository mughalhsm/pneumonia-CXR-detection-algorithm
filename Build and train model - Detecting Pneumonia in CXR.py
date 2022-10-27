#!/usr/bin/env python
# coding: utf-8

# ## Skeleton Code
# 
# The code below provides a skeleton for the model building & training component of your project. You can add/remove/build on code however you see fit, this is meant as a starting point.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from itertools import chain
import sklearn.model_selection
from random import sample 
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, plot_precision_recall_curve, f1_score, confusion_matrix
from tensorflow import keras


# ## Do some early processing of your metadata for easier model training:

# In[2]:


## Load the NIH data to all_xray_df
all_xray_df = pd.read_csv('/data/Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in 
                   glob(os.path.join('/data','images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.sample(3)


# In[3]:


# Replacing all ages above 100 - errouneous data.
all_xray_df.replace(all_xray_df[all_xray_df['Patient Age']>100]['Patient Age'].values,np.nan, inplace = True)


# In[4]:


# Binary indicators of certain diseases 
def split_labels(df):
    labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
    for i in labels:
        df[i] = df['Finding Labels'].map(lambda y: 1.0 if i in y else 0)
        
split_labels(all_xray_df)


# In[5]:


all_xray_df.head()


# In[6]:


# Removing Unnamed Column
all_xray_df.drop('Unnamed: 11', axis = 1, inplace = True)
all_xray_df.head()


# In[7]:


# New column called 'Pneumonia Class'
all_xray_df['Pneumonia Class'] = all_xray_df['Pneumonia'].map(lambda x: 'Positive' if x == 1 else 'Negative')
all_xray_df.head()


# ## Create your training and testing data:

# In[8]:


def create_splits(vargs):
    
    # Initial split
    train_data, val_data = sklearn.model_selection.train_test_split(vargs, test_size = 0.2, stratify = vargs['Pneumonia'])

    # Equal number of +/- pneumonia cases in training set
    train_p_inds = train_data[train_data.Pneumonia==1].index.tolist()
    train_np_inds = train_data[train_data.Pneumonia==0].index.tolist()

    train_np_sample = sample(train_np_inds,len(train_p_inds))
    # Number of negative pneumonia cases the same as number of postive cases for trianing set. 
    train_data = train_data.loc[train_p_inds + train_np_sample]
    # Training data combined to include 50% postive and 50% negative
    

    # % of +ve pneumonia cases in validation set to be equal to natural occurence of the disease in the main dataset
    val_p_inds = val_data[val_data.Pneumonia==1].index.tolist()
    val_np_inds = val_data[val_data.Pneumonia==0].index.tolist()

    # The following code pulls a random sample of non-pneumonia data that's 4 times as big as the pneumonia sample
    val_np_sample = sample(val_np_inds, 4*len(val_p_inds))
    val_data = val_data.loc[val_p_inds + val_np_sample]
    
    return train_data, val_data


# In[9]:


train_data, val_data = create_splits(all_xray_df)


# In[10]:


len(train_data)


# In[11]:


train_data.head()


# In[12]:


len(val_data)


# In[13]:


val_data.head()


# In[14]:


# function to show age distribution
def age(df):
    plt.hist(df['Patient Age'], bins = 10,)
    plt.xlabel('age')
    plt.ylabel('Number of People')
    plt.title('Age Distribution in Dataset')


# In[15]:


age(train_data)


# In[16]:


age(val_data)


# In[17]:


# function to show gender distribution
def gender(df):
    df['Patient Gender'].value_counts().plot(kind='bar')
    plt.xlabel('Gender')
    plt.ylabel('Number of People')
    plt.title('Gender Distribution in Dataset')
    
    return df['Patient Gender'].value_counts()


# In[18]:


gender(train_data)


# In[19]:


gender(val_data)


# In[20]:


# Droping binary value for pneumonia as we have class column. 
train_data.drop('Pneumonia', axis = 1, inplace=True)
val_data.drop('Pneumonia', axis = 1, inplace=True)


# In[21]:


train_data.head()


# # Now we can begin our model-building & training

# #### First suggestion: perform some image augmentation on your data

# In[22]:


def my_image_augmentation():
    
    my_idg = ImageDataGenerator(rescale = 1. / 255.0, 
                                horizontal_flip = True, 
                                vertical_flip = False, 
                                height_shift_range = 0.1, 
                                width_shift_range = 0.1, 
                                rotation_range = 20,
                                shear_range = 0.1, 
                                zoom_range = 0.1)
    
    
    return my_idg

#function to normalize images in validation dataset
def my_image_val_normalize():
    
    my_idg = ImageDataGenerator(rescale = 1. / 255.0, horizontal_flip = False, vertical_flip = False)
    
    
    return my_idg


def make_train_gen(vargs):
    
    ## Create the actual generators using the output of my_image_augmentation for 
    ## your training data
    ## This generator uses a batch size of 32
    idg = my_image_augmentation()
    train_gen = idg.flow_from_dataframe(dataframe=vargs, 
                                        directory=None, 
                                        x_col = 'path',
                                        y_col = 'Pneumonia Class',
                                        class_mode = 'binary',
                                        target_size = (224,224), 
                                        batch_size = 32)
    

    return train_gen

# Generator for the validation dataset
def make_val_gen(vargs):
    
    idg = my_image_val_normalize()
    val_gen = idg.flow_from_dataframe(dataframe=vargs, directory=None, x_col = 'path',y_col = 'Pneumonia Class',class_mode = 'binary',target_size = (224,224), batch_size = 256)
    
    return val_gen


# In[23]:


# Create the augmented training dataset with a batch size of 32
train_gen = make_train_gen(train_data)


# In[24]:


#Create normalized validation dataset
val_gen = make_val_gen(val_data)


# In[25]:


## May want to pull a single large batch of random validation data for testing after each epoch:
valX, valY = val_gen.next()


# In[26]:


## May want to look at some examples of our augmented training data. 
## This is helpful for understanding the extent to which data is being manipulated prior to training, 
## and can be compared with how the raw data look prior to augmentation

t_x, t_y = next(train_gen)
fig, m_axs = plt.subplots(4, 4, figsize = (16, 16))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    if c_y == 1: 
        c_ax.set_title('Pneumonia')
    else:
        c_ax.set_title('No Pneumonia')
    c_ax.axis('off')


# ## Build your model: 
# 
# Recommendation here to use a pre-trained network downloaded from Keras for fine-tuning

# In[27]:


#Loads the VGG 16 model and freezes all but the last CNN layer, returns the new model
def load_pretrained_model():
    
    model = VGG16(include_top=True, weights='imagenet')
    transfer_layer = model.get_layer('block5_pool')
    vgg_model = Model(inputs = model.input, outputs = transfer_layer.output)
    for layer in vgg_model.layers[0:17]:
        layer.trainable = False
    
    
    
    return vgg_model


# In[28]:


#This model has 4 Dense layers and uses a dropout of 0.5 and is added on top of the 
#VGG 16 model with all but its last CNN layer frozen


def build_my_model(vgg_model):
    
    my_model = Sequential()
    my_model.add(vgg_model)
    my_model.add(Flatten())
    my_model.add(Dropout(0.5))
    my_model.add(Dense(1024, activation = 'relu'))
    my_model.add(Dropout(0.5))
    my_model.add(Dense(512, activation = 'relu'))
    my_model.add(Dropout(0.5))
    my_model.add(Dense(256, activation = 'relu'))
    my_model.add(Dense(1, activation = 'sigmoid'))
    

    
    optimizer = Adam(lr = .0001)
    loss = 'binary_crossentropy'
    metrics = ['binary_accuracy']
    
    my_model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

    
  
    
    return my_model


# In[29]:


weight_path="my_model-{epoch:02d}-{val_loss:.2f}.hdf5"

checkpoint = ModelCheckpoint(weight_path, monitor= 'val_loss', verbose=1, save_best_only=True, mode= 'min', save_weights_only = True)

early = EarlyStopping(monitor= 'val_loss', mode= 'min', patience=7)

callbacks_list = [checkpoint, early]


# ### Start training! 

# In[30]:


#loading a pretrained Vgg 16 model with all but its last CNN layer frozen
vgg_model = load_pretrained_model()


# In[31]:


## train your model

# Todo

# history = my_model.fit_generator(train_gen, 
#                           validation_data = (valX, valY), 
#                           epochs = , 
#                           callbacks = callbacks_list)

my_model = build_my_model(vgg_model)

history = my_model.fit_generator(train_gen, validation_data = (valX, valY),  epochs = 10,  callbacks = callbacks_list)


# ##### After training for some time, look at the performance of your model by plotting some performance statistics:
# 
# Note, these figures will come in handy for your FDA documentation later in the project

# In[32]:


def plot_auc(t_y, p_y):
    
    fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
    fpr, tpr, thresholds = roc_curve(t_y, p_y)
    c_ax.plot(fpr, tpr, label = '%s (AUC:%0.2f)'  % ('Pneumonia', auc(fpr, tpr)))
    c_ax.legend()
    c_ax.set_xlabel('False Positive Rate')
    c_ax.set_ylabel('True Positive Rate')
    
    return fpr, tpr, thresholds
    
    
def plot_precision_recall_curve(t_y, p_y):
    fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
    precision, recall, thresholds = precision_recall_curve(t_y, p_y)
    c_ax.plot(recall, precision, label = '%s (AP Score:%0.2f)'  % ('Pneumonia', average_precision_score(t_y,p_y)))
    c_ax.legend()
    c_ax.set_xlabel('Recall')
    c_ax.set_ylabel('Precision')
    
    return precision, recall, thresholds
 
def plot_f1_threshold(t_y, p_y):
    fig, c_ax = plt.subplots(1,1, figsize = (9, 9))
    precision, recall, thresholds = precision_recall_curve(t_y, p_y)
    f1 = 2*(precision*recall)/(precision+recall)
    c_ax.plot(f1[0:len(thresholds)], thresholds, label = 'F1 vs Thresholds')
    c_ax.legend()
    c_ax.set_xlabel('Threshold')
    c_ax.set_ylabel('F1')
    
    

    
# function to calculate the F1 score
def calc_f1(prec,recall):
    
    return (2*(prec*recall)/(prec+recall))

def plot_history(history):
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.plot(history.history["binary_accuracy"], label="train_acc")
    plt.plot(history.history["val_binary_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    


# In[34]:


## plot figures
plot_history(history)
history.history['loss']


# In[35]:


my_model.save('my_model.hdf5')


# In[36]:


my_model = keras.models.load_model('my_model.hdf5')


# In[37]:


my_model_final = keras.models.load_model('my_model.hdf5')


# In[38]:


my_model_final.summary()


# In[39]:


my_model.load_weights('my_model-02-0.54.hdf5')
pred_Y = my_model.predict(valX, batch_size = 32, verbose = True)


# In[40]:


fpr, tpr, thresholds = plot_auc(valY, pred_Y)


# In[41]:


prec, recall, threshold_2 = plot_precision_recall_curve(valY, pred_Y)


# In[42]:


plot_f1_threshold(valY, pred_Y)


# Once you feel you are done training, you'll need to decide the proper classification threshold that optimizes your model's performance for a given metric (e.g. accuracy, F1, precision, etc.  You decide) 

# In[43]:


# Threshold of 0.35 will result in F1 score of 0.32
threshold_select = 0.35
index = (np.abs(threshold_2 - threshold_select)).argmin()
precision = prec[index]
rec = recall[index]


# In[44]:


f1_score = calc_f1(precision,rec)


# In[45]:


print('F1 Score is: {}'.format(f1_score))
print('Precisions is: {}'.format(precision))
print('Recall is: {}'.format(rec)) 


# In[46]:


## Let's look at some examples of true vs. predicted with our best model: 

# Todo

fig, m_axs = plt.subplots(10, 10, figsize = (16, 16))
i = 0
for (c_x, c_y, c_ax) in zip(valX[0:100], valY[0:100], m_axs.flatten()):
    c_ax.imshow(c_x[:,:,0], cmap = 'bone')
    if c_y == 1: 
        if pred_Y[i] > .35:
            c_ax.set_title('1, 1')
        else:
            c_ax.set_title('1, 0')
    else:
        if pred_Y[i] > .35: 
            c_ax.set_title('0, 1')
        else:
            c_ax.set_title('0, 0')
    c_ax.axis('off')
    i=i+1


# In[47]:


## Just save model architecture to a .json:

model_json = my_model.to_json()
with open("my_model.json", "w") as json_file:
    json_file.write(model_json)


# In[ ]:




