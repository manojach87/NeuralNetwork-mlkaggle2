#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 18:16:56 2019

@author: manoj
"""


import pandas as pd  
import numpy as np
import os
import time
from sklearn.svm import SVC 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score  
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA



def read_file(file_name):
    if(os.path.exists(file_name)):
        return pd.read_csv(file_name, encoding='latin1')
        

trainx = read_file("./train_X.csv")
trainy = read_file("./train_Y.csv")
testx  = read_file("./test_X.csv")

def visuals(trainx):
    import seaborn as sns
    import matplotlib.pyplot as plt
    corr_mat = trainx.corr()
    plt.figure(figsize=(12,12))

    sns.heatmap(corr_mat,annot=False,cmap="RdYlGn")
   
    f, axi = plt.subplots(figsize=(10,4))
    for i in range (1,54):
        plt.scatter(y=trainy["t1"],x=trainx['f'+str(i)],cmap='jet')
    plt.show()

    sns.pairplot(testx,size=2.5)
    plt.show()
    
visuals(trainx)