# -*- coding: utf-8 -*-
"""Neural_Network_Model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16xXJpw4-8VhnaNPBUvbaQsin5k5e-xxw
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
now = datetime.now()
global epochs
trainings=5000
checkpoint_interval=1
epochs = 1
tf.set_random_seed(0)
model_filepath='./model.save'

import zipfile
zip_ref = zipfile.ZipFile('./train_Y.zip', 'r')
zip_ref.extractall('./')
zip_ref.close()

X = pd.read_csv("./train_X.csv")

Y = pd.read_csv("./train_Y.csv")

pred = pd.read_csv('./test_X.csv')

X.set_index("idx", inplace=True)
Y.set_index('idx', inplace=True)
pred.set_index('idx', inplace=True)

# Normalize Data
Xnames = X.columns
prednames = pred.columns
X = tf.keras.utils.normalize(X.values)
pred = tf.keras.utils.normalize(pred.values)
Y=Y.values

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

y_train=to_categorical(y_train)
y_test=to_categorical(y_test)

y_test.shape[1]

#KNN can't read panda df's so need to convert all df's 
#to values only by removing headers


y_trainV = y_train
y_testV = y_test
X_testV = X_test
X_trainV = X_train
predV = pred

Y_train_1h = y_trainV
Y_test_1h = y_testV

def buildNN():
    global model
    model = Sequential()
    model.add(Dense (108, input_dim = 54, activation = 'relu'))
    model.add(Dense (500, activation = 'relu'))
    model.add(Dense (500, activation = 'relu'))
    model.add(Dense (500, activation = 'relu'))
    #model.add(Dense (72, activation = 'relu'))
    #model.add(Dense (64, activation = 'relu'))
    #model.add(Dense (48, activation = 'relu'))
    #model.add(Dense (32, activation = 'relu'))
    #model.add(Dense (16, activation = 'relu'))
    model.add(Dense (8,   activation = 'softmax'))
    #model.compile(loss = tf.losses.softmax_cross_entropy,
    #              optimizer = 'adam',
    #              metrics = ['accuracy']
    #              #,steps_per_epoch=1
    #              )
    model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              #loss='poisson',
              #loss='kullback_leibler_divergence',
              #loss='cosine_proximity',
              #loss='categorical_crossentropy',
              metrics=['accuracy'])



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
    df.to_csv(str(epochs) + 'Epochs_' + now.strftime('%b%w_%H,%M') + '.csv', index_label='idx')

buildNN()

model=tf.keras.models.load_model(model_filepath)

#model.fit(X_trainV, Y_train_1h, epochs=epochs, callbacks=[cp_callback])
for i in range(trainings):
  print("Training "+str(i+1)+"/"+str(trainings))
  model.fit(X_trainV, Y_train_1h, epochs=epochs)
  if(i%checkpoint_interval==checkpoint_interval-1):
    tf.keras.models.save_model(model,model_filepath)
    print("Model from training "+str(i+1)+" saved!" )

evaluateModel()

writeResults()

def buildNN():
    global model
    model = Sequential()
    model.add(Dense(53, activation='relu', input_dim = 53))
    model.add(Dropout(0.3))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(8, activation='softmax'))
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

