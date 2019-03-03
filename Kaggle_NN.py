#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 12:29:39 2019

@author: manoj
"""

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
epochs = 5

def loadData():
    global X 
    X = pd.read_csv("train_X.csv")
    global Y 
    Y = pd.read_csv("train_Y.csv")
    global pred
    pred = pd.read_csv('test_X.csv')
    X.set_index("idx", inplace=True)
    Y.set_index('idx', inplace=True)
    pred.set_index('idx', inplace=True)
    
def splitTestTrain():
    global X
    global pred
    X = tf.keras.utils.normalize(X)
    pred = tf.keras.utils.normalize(pred)
    global y_train
    global y_test
    global X_test
    global X_train
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
    
#KNN can't read panda df's so need to convert all df's 
#to values only by removing headers
def selectValues():
    global y_trainV
    global y_testV
    global X_testV
    global X_trainV
    global predV
    y_trainV = y_train.values
    y_testV = y_test.values
    X_testV = X_test.values
    X_trainV = X_train.values
    predV = pred.values

def oneHotEncode():
    global Y_train_1h
    global Y_test_1h
    Y_train_1h = to_categorical(y_trainV)
    Y_test_1h = to_categorical(y_testV)

def buildNN():
    global model
    model = Sequential()
    model.add(Dense (54, activation = 'relu'))
    model.add(Dense (500, activation = 'relu'))
    model.add(Dense (200, activation = 'tanh'))
    model.add(Dense (8,   activation = 'softmax'))
    model.compile(loss = tf.losses.softmax_cross_entropy,
                  optimizer = 'adam',
                  metrics = ['accuracy'],
                  steps_per_epoch = 10)
    return model

def fitModel(epochs):
    model.fit(X_trainV, Y_train_1h, epochs=epochs)

def evaluateModel():
    global Y_pred
    global score
    print(model.evaluate(X_testV, Y_test_1h)) 
    Y_pred = model.predict(pred)

def writeResults():
    global df
    df = DataFrame({'t1': np.argmax(Y_pred, axis=1)})
    df.index = np.arange(1, len(Y_pred) + 1)
    df.to_csv(str(epochs) + 'Epochs' + '_.csv', index_label='idx')

loadData()

splitTestTrain()

selectValues()

oneHotEncode()

buildNN()

fitModel(epochs)

evaluateModel()

writeResults()


