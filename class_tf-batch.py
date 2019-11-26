#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 15:48:24 2019

@author: robabbott
"""
#%%
import numpy as np
import tensorflow as tf
import time

from pandas import DataFrame

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
#%%

from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LeakyReLU
from tensorflow.keras.utils import to_categorical

#%%
CLASS_SEP = 1
FLIP_Y = 0
#%%
X, Y = make_classification(100, n_features=50, n_classes=3, 
                           n_clusters_per_class=1, n_redundant=0, 
                           class_sep=CLASS_SEP, flip_y=FLIP_Y, random_state=87)
#%%
X = tf.keras.utils.normalize(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, random_state=42)
#%%
Y_train_1h = to_categorical(Y_train)
Y_test_1h = to_categorical(Y_test)
#%%
model = Sequential()
model.add(Dense(50, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
#%%
model.compile(loss=tf.losses.mean_squared_error,
              optimizer='adam',
              metrics=['accuracy'],
              steps_per_epoch=1)
#%%
stime = time.monotonic()
model.fit(X_train, Y_train_1h, batch_size=len(X_train), epochs=10000, verbose=0)
etime = time.monotonic()
#%%
print('training time (s): ', etime-stime)

