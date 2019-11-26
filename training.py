#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:05:30 2019

@author: bburke1
"""
#%%
global pd
import pandas as pd
import numpy as np
import tensorflow as tf    
from pandas import DataFrame    
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_classification
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LeakyReLU
from tensorflow.keras.utils import to_categorical
from datetime import datetime
#%%
now = datetime.now()
global epochs
epochs = 10
#%%
# Read X
X = pd.read_csv("train_X.csv")
#%%
OX=X
#%%
X=OX
#%%
# Read Y
Y = pd.read_csv("train_Y.csv")
#%%
# Read Testing Data
pred = pd.read_csv('test_X.csv')
#%%
X.set_index("idx", inplace=True)
#%%
Y.set_index('idx', inplace=True)
pred.set_index('idx', inplace=True)

#%%
# Normalize Data
from sklearn import preprocessing
Xnames = X.columns
prednames = pred.columns
#%%
X = tf.keras.utils.normalize(X.values)
pred = tf.keras.utils.normalize(pred.values)
#%%
#Norm_X = preprocessing.normalize(X)
#Norm_pred = preprocessing.normalize(pred)

#%%
#np.savetxt("foo.csv", Norm_pred, delimiter=",")
#%%

#%%
#X= pd.DataFrame(Norm_X, columns=Xnames, dtype=np.float)
#pred=pd.DataFrame(Norm_pred, columns=prednames, dtype=np.float)
#%%
#X=Norm_X
#pred=Norm_pred
#%%

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

#%%

#%%
#KNN can't read panda df's so need to convert all df's 
#to values only by removing headers


y_trainV = y_train
y_testV = y_test
X_testV = X_test
X_trainV = X_train
predV = pred


#%%
#def oneHotEncode():
#    global Y_train_1h
#    global Y_test_1h
#    Y_train_1h = to_categorical(y_trainV)
#    Y_test_1h = to_categorical(y_testV)
#%%
#Y_train_1h = to_categorical(y_trainV)
#Y_test_1h = to_categorical(y_testV)
Y_train_1h = y_trainV
Y_test_1h = y_testV

#%%
def buildNN():
    global model
    model = Sequential()
    model.add(Dense (108, input_dim = 54, activation = 'relu'))
    model.add(Dense (100, activation = 'relu'))
    model.add(Dense (50, activation = 'tanh'))
    model.add(Dense (8,   activation = 'softmax'))
    model.compile(loss = tf.losses.softmax_cross_entropy,
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    return model
#%%
def fitModel(epochs):
    model.fit(X_trainV, Y_train_1h, epochs=epochs)
#%%
def evaluateModel():
    global Y_pred
    global score
    print(model.evaluate(X_testV, Y_test_1h)) 
    Y_pred = model.predict(pred)
#%%
def writeResults():
    global df
    df = DataFrame({'t1': np.argmax(Y_pred, axis=1)})
    df.index = np.arange(1, len(Y_pred) + 1)
    df.to_csv(str(epochs) + 'Epochs_' + now.strftime('%b%w_%H,%M') + '.csv', index_label='idx')

#%%

buildNN()
#%%
model.fit(X_trainV, Y_train_1h, epochs=epochs)

#%%
fitModel(epochs)
#%%

evaluateModel()
#%%

writeResults()