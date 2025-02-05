#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
import matplotlib.pyplot as plt


# In[16]:


#loading data
num_classes=10
data=keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=data.load_data()


# In[17]:


"""normalization [0,1]"""
x_train=keras.utils.normalize(x_train,axis=1)
x_test=keras.utils.normalize(x_test,axis=1)
print(x_train[23][23])


# In[18]:


model=Sequential(
    [
        Flatten(),
        Dense(128,activation='relu'),
        Dense(128,activation='relu'),
        Dense(num_classes,activation='softmax')
    ]
)


# In[19]:


model.compile(optimizer="Adam",loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[22]:


#train a model
model.fit(x_train,y_train,epochs=3)


# In[23]:


loss,acc=model.evaluate(x_test,y_test)
print(loss)
print(acc)


# In[26]:


model.save('handwritten_digits_recognition.keras')


# In[ ]:




