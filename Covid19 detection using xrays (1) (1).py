#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import os
import shutil


# In[22]:


#create data for covid positive samples

FILE_PATH = r'C:\Users\ASUS\Desktop\COVID DETECTION USING XARAY\covid-chestxray-dataset-master/metadata.csv'
IMAGES_PATH = r'C:\Users\ASUS\Desktop\COVID DETECTION USING XARAY\covid-chestxray-dataset-master/images'


# In[23]:


df = pd.read_csv(FILE_PATH)
print(df.shape)


# In[24]:


df.head(50)


# # extracting xray images of covid patients

# In[25]:


TARGET_DIR = r'C:\Users\ASUS\Covid 19 detection/Covid Dataset'

if not os.path.exists(TARGET_DIR):
    os.mkdir(TARGET_DIR)
    print("Covid dataset created")


# In[26]:


df['finding'].value_counts()


# In[27]:


df['RT_PCR_positive'].value_counts()


# In[ ]:





# In[28]:


print(df.dtypes)
df['view'].value_counts()


# # condition for extracting covid patient xray images

# In[29]:


cnt = 0
for(i,row) in df.iterrows():
    if row['finding']=="Pneumonia/Viral/COVID-19"and row['RT_PCR_positive']=="Y" and row['view']=='PA':
        filename = row['filename']
        image_path = os.path.join(IMAGES_PATH,filename) 
        image_copy_path = os.path.join(TARGET_DIR,filename)
        shutil.copy2(image_path,image_copy_path)
        print("Moving image ",cnt)
        cnt = cnt + 1
        
print("Total number of images is ",cnt)


# # Extracting normal patient xrays

# In[30]:


#Sampling of images from Kaggle dataset

import random
KAGGLE_FILE_PATH = r'C:\Users\ASUS\Desktop\COVID DETECTION USING XARAY\chest_xray\train\NORMAL'
TARGET_NORMAL_DIRECTORY =  r'C:\Users\ASUS\Covid 19 detection/Normal Dataset'

random.seed(3)
image_names = os.listdir(KAGGLE_FILE_PATH)
image_names


# In[31]:


# random shuffling of images

random.shuffle(image_names)
for i in range(110):
    image_name = image_names[i]
    image_path = os.path.join(KAGGLE_FILE_PATH,image_name)
    target_path = os.path.join(TARGET_NORMAL_DIRECTORY,image_name)
    shutil.copy2(image_path,target_path)
    print("Copying image",i)


# # TRAIN TEST SPLIT

# In[32]:


COVID_DATASET_PATH =  r'C:\Users\ASUS\Covid 19 detection\Covid Dataset'
NORMAL_DATASET_PATH = r'C:\Users\ASUS\Covid 19 detection\Normal Dataset'
covid_image_names = os.listdir(COVID_DATASET_PATH)
normal_image_names = os.listdir(NORMAL_DATASET_PATH)

TRAIN_NORMAL_PATH = r'C:\Users\ASUS\Covid 19 detection\Train\normal'
TRAIN_COVID_PATH =  r'C:\Users\ASUS\Covid 19 detection\Train\covid'

VAL_NORMAL_PATH = r'C:\Users\ASUS\Covid 19 detection\Test\normal'
VAL_COVID_PATH = r'C:\Users\ASUS\Covid 19 detection\Test\covid'

for i in range(80):
    cvd_img_nms = covid_image_names[i]
    nrml_img_nms = normal_image_names[i]
    COVID_IMG_PATH = os.path.join(COVID_DATASET_PATH,cvd_img_nms)
    TARGET_IMG_PATH = os.path.join(TRAIN_COVID_PATH,cvd_img_nms)
    shutil.copy2(COVID_IMG_PATH,TARGET_IMG_PATH)
    
    NORMAL_IMG_PATH = os.path.join(NORMAL_DATASET_PATH,nrml_img_nms)
    TARGET2_IMG_PATH = os.path.join(TRAIN_NORMAL_PATH,nrml_img_nms)
    shutil.copy2(NORMAL_IMG_PATH,TARGET2_IMG_PATH)
    print("Copying image",i)
for i in range(80,110):
    cvd2_img_nms = covid_image_names[i]
    nrml2_img_nms = normal_image_names[i]
    COVID2_IMG_PATH = os.path.join(COVID_DATASET_PATH,cvd2_img_nms)
    TARGET3_IMG_PATH = os.path.join(VAL_COVID_PATH,cvd2_img_nms)
    shutil.copy2(COVID2_IMG_PATH,TARGET3_IMG_PATH)
    
    NORMAL2_IMG_PATH = os.path.join(NORMAL_DATASET_PATH,nrml2_img_nms)
    TARGET4_IMG_PATH = os.path.join(VAL_NORMAL_PATH,nrml2_img_nms)
    shutil.copy2(NORMAL2_IMG_PATH,TARGET4_IMG_PATH)
    print("Copying VALIDATION image",i)


# # model building

# In[33]:


import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image


# In[34]:


# this will be a CNN(convolutional neural network) based deep learning model in Keras
# two parts: feature extraction and classification
# for feature extraction we will apply 2D filters over the image
# the number of filters will be increased in every next layer (this is a layered architecture)
# in total there will be 3-4 convolutional layers for feature extraction and then one layer for classification

model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])


# In[35]:


model.summary()


# In[36]:


# Train Data Generator
train_datagen = image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True)
test_datagen = image.ImageDataGenerator(1./255)

train_generator = train_datagen.flow_from_directory(
        r'C:\Users\ASUS\Covid 19 detection\Train',
        target_size =(224,224),
        batch_size = 32,
        class_mode = 'binary')



# In[37]:


train_generator.class_indices


# In[38]:


validation_generator = test_datagen.flow_from_directory(
         r'C:\Users\ASUS\Covid 19 detection\test',
        target_size =(224,224),
        batch_size = 32,
        class_mode = 'binary')


# In[39]:


validation_generator.class_indices


# # Training the model

# In[ ]:


hist = model.fit_generator(
        train_generator,
        steps_per_epoch=8,
        epochs=10,
        validation_data = validation_generator,
        validation_steps=2)


# In[ ]:




