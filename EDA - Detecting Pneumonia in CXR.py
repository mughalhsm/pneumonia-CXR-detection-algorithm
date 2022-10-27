#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from glob import glob
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import chain
from skimage import io
import scipy.stats

##Import any other packages you may need here:


# EDA is open-ended, and it is up to you to decide how to look at different ways to slice and dice your data. A good starting point is to look at the requirements for the FDA documentation in the final part of this project to guide (some) of the analyses you do. 
# 
# This EDA should also help to inform you of how pneumonia looks in the wild. E.g. what other types of diseases it's commonly found with, how often it is found, what ages it affects, etc. 
# 
# Note that this NIH dataset was not specifically acquired for pneumonia. So, while this is a representation of 'pneumonia in the wild,' the prevalence of pneumonia may be different if you were to take only chest x-rays that were acquired in an ER setting with suspicion of pneumonia. 

# Perform the following EDA:
# * The patient demographic data such as gender, age, patient position,etc. (as it is available)
# * The x-ray views taken (i.e. view position)
# * The number of cases including: 
#     * number of pneumonia cases,
#     * number of non-pneumonia cases
# * The distribution of other diseases that are comorbid with pneumonia
# * Number of disease per patient 
# * Pixel-level assessments of the imaging data for healthy & disease states of interest (e.g. histograms of intensity values) and compare distributions across diseases.
# 
# Note: use full NIH data to perform the first a few EDA items and use `sample_labels.csv` for the pixel-level assassements. 

# Also, **describe your findings and how will you set up the model training based on the findings.**

# In[2]:


## All NIH Data
all_xray_df = pd.read_csv('/data/Data_Entry_2017.csv')
all_xray_df.sample(5)


# In[3]:


len(all_xray_df)


# In[4]:


## Load 'sample_labels.csv' data for pixel level assessments
sample_df = pd.read_csv('sample_labels.csv')
sample_df.sample(5)


# In[5]:


len(sample_df)


# # Cleaning the data. 
# 1. Delete irrelevant data - last four columns 
# 2. Clean sample data - remove 'Y' and '0' from the data. 
# 3. Seperate the findings labels into a binary flag. 
# 

# In[6]:


all_xray_df.head(40)
all_xray_df.head(50)


# In[7]:


for col in all_xray_df.columns:
    print(col)


# In[8]:


all_xray_df2 = all_xray_df.drop(columns=['OriginalImage[Width', 'Height]', 'OriginalImagePixelSpacing[x', 'y]', 'Unnamed: 11'])


# In[9]:


all_xray_df2.head()
# new data set with clinically or algorithm relevant data. 


# In[10]:


sample_df.head()


# In[11]:


sample_df2 = sample_df.drop(columns=['OriginalImageWidth', 'OriginalImageHeight', 'OriginalImagePixelSpacing_x', 'OriginalImagePixelSpacing_y'])


# In[12]:


sample_df2.head()


# In[13]:


sample_df2['Patient Age'].replace(r'^(0+)', '', inplace=True, regex=True)
## replace zero from age value.


# In[14]:


sample_df2['Patient Age'].replace(r'Y', '', inplace=True, regex=True)
## replace Y from age value. 


# In[15]:


sample_df2.head(30)


# In[16]:


from itertools import chain


# In[17]:


all_labels = np.unique(list(chain(*sample_df2['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels = [x for x in all_labels if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels), all_labels))
for c_label in all_labels:
    if len(c_label)>1: # leave out empty labels
        sample_df2[c_label] = sample_df2['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
sample_df2.sample(3)


# In[18]:


sample_df2.head(10)
# I have seperated the Findings Labels columns with a binary flag - as well as keeping the 'Finding Labels' column. 
# I will do the same for all_xray_data.
# I did consider at this moment deleting the hernia column as it not relevant for chest xray findings for 
# pneumonia in my clinical opinion.


# In[19]:


all_labels_xray = np.unique(list(chain(*all_xray_df2['Finding Labels'].map(lambda x: x.split('|')).tolist())))
all_labels_xray = [x for x in all_labels_xray if len(x)>0]
print('All Labels ({}): {}'.format(len(all_labels_xray), all_labels_xray))
for c_label in all_labels_xray:
    if len(c_label)>1: # leave out empty labels
        all_xray_df2[c_label] = all_xray_df2['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)
all_xray_df2.sample(3)


# In[20]:


all_xray_df2.head(20)


# # Visualization of the metadata
# 1. Distribution of basic demographics (age, gender, xray view)
# 1. Distribution of disease and comorbidities in the dataset.
# 3. Conclusions on how to set up my training model. 

# In[21]:


plt.figure(figsize=(10,6))
plt.hist(all_xray_df2['Patient Age'])


# In[22]:


all_xray_df2['Patient Age'].max()
## Oldest patient age is 414 ?? - Error - delete the sample - will remove sample from file when Age > 120.


# In[23]:


all_xray_df['Patient Age'].min()
## Youngest patient in xray data is 1. 


# In[24]:


sample_df2["Patient Age"].max()
# 94 - output in '' - is this string or number?


# In[25]:


sample_df2['Patient Age'].min()
## 10 


# In[26]:


testdf = all_xray_df.copy()


# In[27]:


testdf
testdf.drop(testdf[testdf['Patient Age'] >= 120].index, inplace = True)


# In[28]:


len(all_xray_df)


# In[29]:


len(testdf)


# In[30]:


all_xray_df2.drop(all_xray_df2[all_xray_df2['Patient Age'] >= 120].index, inplace = True)
all_xray_df2.drop(all_xray_df2[all_xray_df2['Patient Age'] < 18].index, inplace = True)
# # Remove data to inculde only adults aged 18 and below 120 years of age. .


# In[31]:


len(all_xray_df2)


# # Distribution of Age in NIH Chest X-Ray Data for Adults

# In[32]:


plt.figure(figsize=(12,6))
plt.hist(all_xray_df2['Patient Age'])
plt.title('Distribution of Age in NIH Chest X-Ray Data for Adults')
plt.xlabel('Age (years)')
plt.ylabel('Population')
# Age distribution is much cleaner - removed 
# Potentially could remove all data to inculde only adults aged 18 or above. 


# In[33]:


plt.figure(figsize=(12,6))
plt.hist(sample_df2['Patient Age'])


# In[34]:


sample_df2['Patient Age'].replace(r'D', '99', inplace=True, regex=True)
sample_df2['Patient Age'].replace(r'M', '99', inplace=True, regex=True)
sample_df2['Patient Age'] = pd.to_numeric(sample_df2['Patient Age'])
sample_df2.drop(sample_df2[sample_df2['Patient Age'] >= 120].index, inplace = True)
sample_df2.drop(sample_df2[sample_df2['Patient Age'] < 18].index, inplace = True)


# # Distribution of Age in Sample Data for Adults

# In[35]:


plt.figure(figsize=(12,6))
plt.hist(sample_df2['Patient Age'])
plt.title('Distribution of Age in Sample Data for Adults')
plt.xlabel('Age (years)')
plt.ylabel('Population')


# # Distribution of Gender in NIH Chest X-Ray Data for Adults

# In[36]:


plt.figure(figsize=(6,6))
all_xray_df2['Patient Gender'].value_counts().plot(kind='bar')
plt.title('Distribution of Gender in NIH Chest X-Ray Data for Adults')
plt.xlabel('Gender')
plt.ylabel('Population')


# # Distribution of Gender in Sample Data for Adults

# In[37]:


plt.figure(figsize=(6,6))
sample_df2['Patient Gender'].value_counts().plot(kind='bar')
plt.title('Distribution of Gender in Sample Data for Adults')
plt.xlabel('Gender')
plt.ylabel('Population')


# # Distribution of View Postion of CXR in NIH & Sample Data

# In[38]:


plt.figure(figsize=(6,6))
all_xray_df2['View Position'].value_counts().plot(kind='bar')
plt.title('Distribution of Xray View Position in NIH Chest X-Ray Data for Adults')
plt.xlabel('View')
plt.ylabel('Population')


# In[39]:


plt.figure(figsize=(6,6))
sample_df2['View Position'].value_counts().plot(kind='bar')
plt.title('Distribution of Xray View Position in Sample Data for Adults')
plt.xlabel('View')
plt.ylabel('Population')


# # Distribution of Disease

# In[40]:


ax = all_xray_df2[all_labels_xray].sum().plot(kind='bar')
ax.set(ylabel = 'Number of Images with Label')
plt.title('Distribution of Co-morbidities')


# In[41]:


bx = sample_df2[all_labels_xray].sum().plot(kind='bar')
ax.set(ylabel = 'Number of Images with Label')
plt.title('Distribution of Co-morbidities - Sample Population')


# In[42]:


plt.figure(figsize=(16,6))
all_xray_df2[all_xray_df2.Pneumonia==1]['Finding Labels'].value_counts()[0:30].plot(kind='bar')


# In[43]:


plt.figure(figsize=(16,6))
sample_df2[sample_df2.Pneumonia==1]['Finding Labels'].value_counts()[0:30].plot(kind='bar')


# # Conclusions
# As mentioned above I have cleaned the data to include only adults and have removed errouneous results from the sample data set.

# 

# # Displaying Sample Data - Pixel Level Data

# In[44]:


sample_df2.shape


# In[45]:


sample_df2.head()


# In[46]:


# Image path for images in sample data - this is reading all the image
# paths in the sample data. Images are stored in the data/images folder
image_path = [glob(f'/data/images*/*/{i}')[0] for i in sample_df2['Image Index'].values]


# In[47]:


# First 20 CXRs in the sample data. 
fig, m_axs = plt.subplots(5,4, figsize = (16, 16))
m_axs = m_axs.flatten()
imgs = image_path


for img, ax in zip(imgs, m_axs):
    img = io.imread(img)
    ax.imshow(img,cmap='gray')


# In[48]:


#display 20 imgs that have pneumonia
fig, m_axs = plt.subplots(5,4, figsize = (16, 16))
m_axs = m_axs.flatten()
pneumonia_imgs = sample_df2[sample_df2.Pneumonia==1]['Image Index']
pneumonia_image_path = [glob(f'/data/images*/*/{i}')[0] for i in pneumonia_imgs]

for img, ax in zip(pneumonia_image_path, m_axs):
    img = io.imread(img)
    ax.imshow(img,cmap='gray')


# In[49]:


#display 20 imgs that have no findings
fig, m_axs = plt.subplots(5,4, figsize = (16, 16))
m_axs = m_axs.flatten()
normal_CXR = sample_df2[sample_df2['No Finding']==1]['Image Index']
normal_CXR_path = [glob(f'/data/images*/*/{i}')[0] for i in normal_CXR]

for img, ax in zip(normal_CXR_path, m_axs):
    img = io.imread(img)
    ax.imshow(img,cmap='gray')


# In[50]:


#display 20 imgs that have hernia
fig, m_axs = plt.subplots(5,4, figsize = (16, 16))
m_axs = m_axs.flatten()
hernia_CXR = sample_df2[sample_df2['Hernia']==1]['Image Index']
hernia_CXR_path = [glob(f'/data/images*/*/{i}')[0] for i in hernia_CXR]

for img, ax in zip(hernia_CXR_path, m_axs):
    img = io.imread(img)
    ax.imshow(img,cmap='gray')


# In[51]:


#Generating img paths for pneumonia and diseases mose frequently occuring with pneumonia but without pneumonia.
infiltration_imgs = sample_df2[(sample_df2.Infiltration==1) & (sample_df2.Pneumonia==0)]['Image Index']
infiltration_image_path = [glob(f'/data/images*/*/{i}')[0] for i in infiltration_imgs]

edema_imgs = sample_df2[(sample_df2.Edema==1) & (sample_df2.Pneumonia==0)]['Image Index']
edema_image_path = [glob(f'/data/images*/*/{i}')[0] for i in edema_imgs]

atelectasis_imgs = sample_df2[(sample_df2.Atelectasis==1) & (sample_df2.Pneumonia==0)]['Image Index']
atelectasis_image_path = [glob(f'/data/images*/*/{i}')[0] for i in atelectasis_imgs]

effusion_imgs = sample_df2[(sample_df2.Effusion==1) & (sample_df2.Pneumonia==0)]['Image Index']
effusion_image_path = [glob(f'/data/images*/*/{i}')[0] for i in effusion_imgs]

consolidation_imgs = sample_df2[(sample_df2.Consolidation==1) & (sample_df2.Pneumonia==0)]['Image Index']
consolidation_image_path = [glob(f'/data/images*/*/{i}')[0] for i in consolidation_imgs]


# In[52]:


#fnc to plot intesity values
def multiple_intense_val(path):
    image = io.imread(path)
    plt.hist(image.ravel(),bins=256)
    plt.legend(['pneumonia', 'infiltration', 'edema', 'atelectasis', 'effusion', 'consolidation'])


# In[53]:


#plotting intensity value of pneumonia and diseases most common with pneumonia(single cases)
multiple_intense_val(pneumonia_image_path[5])
multiple_intense_val(infiltration_image_path[0])
multiple_intense_val(edema_image_path[0])
multiple_intense_val(atelectasis_image_path[0])
multiple_intense_val(effusion_image_path[0])
multiple_intense_val(consolidation_image_path[0])


# In[54]:


#Threshold 20 to remove background pixels. Boundary = 250
thresh = 20
boundary = 250


# In[55]:


# FUnction to plot intensity values. 
def cumulative_intensity(disease_img_path, thresh, boundary):
    img_intensities = []
    for i in disease_img_path: 
        img = io.imread(i)
        
        img_intensities.extend(img[(img > thresh) & (img < boundary)].tolist())
    x = plt.hist(img_intensities,bins=256)
        
    return img_intensities


# In[56]:


pneumonia_intensities = cumulative_intensity(pneumonia_image_path, thresh, boundary)


# In[57]:


infiltration_intensities = cumulative_intensity(infiltration_image_path, thresh, boundary)


# In[58]:


edema_intensities = cumulative_intensity(edema_image_path, thresh, boundary)


# In[59]:


atelectasis_intensities = cumulative_intensity(atelectasis_image_path, thresh, boundary)


# In[60]:


effusion_intensities = cumulative_intensity(effusion_image_path, thresh, boundary)


# In[61]:


consolidation_intensities = cumulative_intensity(consolidation_image_path, thresh, boundary)


# In[62]:


pneumonia_mode = scipy.stats.mode(pneumonia_intensities)[0][0]
infiltration_mode = scipy.stats.mode(infiltration_intensities)[0][0]
edema_mode = scipy.stats.mode(edema_intensities)[0][0]
atelectasis_mode = scipy.stats.mode(atelectasis_intensities)[0][0]
effusion_mode = scipy.stats.mode(effusion_intensities)[0][0]
consolidation_mode = scipy.stats.mode(consolidation_intensities)[0][0]


# In[63]:


print(pneumonia_mode, infiltration_mode, edema_mode, atelectasis_mode, effusion_mode, consolidation_mode)
# these are the modal values of all the types of graphs. 


# In[64]:


#comparing first 10 pneumonia diagnoses intensity mode with the general pneumonia population intensity mode
for path in pneumonia_image_path[0:10]:
    
    img = io.imread(path)
    # test image path into image.
    
    img_mask = (img > thresh) & (img < boundary)
    # test images with threshold (Otsu's Method)
    
    mode = scipy.stats.mode(img[img_mask])[0][0]
    # modal value of this immage - pixel with highest peak on grapth.
    
    pneumonia_delta = np.abs(mode - pneumonia_mode) # for the first image = 17
    infiltration_delta = np.abs(mode - infiltration_mode) 
    edema_delta = np.abs(mode - edema_mode)
    atelectasis_delta = np.abs(mode - atelectasis_mode)
    effusion_delta = np.abs(mode - effusion_mode)
    consolidation_delta = np.abs(mode - consolidation_mode)
    
    min_mode = min([pneumonia_delta, infiltration_delta, edema_delta, atelectasis_delta, effusion_delta, consolidation_delta ])
    
    if min_mode == pneumonia_delta:
        print("Correct Label")
    else:
        print("Incorrect Label")
    print (mode)    


# In[65]:


# 90% accuracy using the above method after data has been cleaned. 


# In[ ]:




