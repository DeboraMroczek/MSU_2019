#!/usr/bin/env python
# coding: utf-8

from tensorflow.python import keras
import tensorflow as tf
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
from os import listdir
from numpy import reshape 
from numpy import array
from sklearn import preprocessing

#Define path for training sets
PATHOLOGICAL_ID = "Pathological-Training/"
ACCEPTABLE_ID = "Non-Pathological-Training"
ACCEPTABLETEST_ID = "AcceptableSets_Test"
PATHOLOGICALTEST_ID = "PathologicalSets_Test"

#Define functions to read files contained in training set
def pathological_path(filename):
    return os.path.join(PATHOLOGICAL_ID, filename)

def acceptable_path(filename):
    return os.path.join(ACCEPTABLE_ID, filename)

def acceptabletest_path(filename):
    return os.path.join(ACCEPTABLETEST_ID, filename)    

def pathologicaltest_path(filename):
    return os.path.join(PATHOLOGICALTEST_ID, filename) 

#Name of the folder for training sets
Names_Pathological = os.listdir('Pathological-Training')
Names_Acceptable = os.listdir('Non-Pathological-Training')
Names_Pathological_Test = os.listdir('PathologicalSets_Test')
Names_Acceptable_Test = os.listdir('AcceptableSets_Test')

#Open first file and define pandas data frame for the pathological portion of the training set 
infile = open(pathological_path(Names_Pathological[0]),'r')
press_pathological = pd.read_csv(infile,sep = '\t',names = ('T','muB','P'))

infile = open(acceptable_path(Names_Acceptable[0]),'r')
press_acceptable = pd.read_csv(infile,sep = '\t',names = ('T','muB','P'))

infile = open(pathologicaltest_path(Names_Pathological_Test[0]),'r')
press_pathological_test = pd.read_csv(infile,sep = '\t',names = ('T','muB','P'))

infile = open(acceptabletest_path(Names_Acceptable_Test[0]),'r')
press_acceptable_test = pd.read_csv(infile,sep = '\t',names = ('T','muB','P'))


#Import training files and define Pandas data frame

for i in range(1,len(Names_Pathological)):
    infile = open(pathological_path(Names_Pathological[i]),'r')
    Press = pd.read_csv(infile,sep = '\t',names = ('T','muB','P'))
    press_pathological =press_pathological.append(pd.DataFrame(Press))
    

for i in range(1,len(Names_Acceptable)):
    infile = open(acceptable_path(Names_Acceptable[i]),'r')
    Press = pd.read_csv(infile,sep = '\t',names = ('T','muB','P'))
    press_acceptable =press_acceptable.append(pd.DataFrame(Press))

for i in range(1,len(Names_Pathological_Test)):
    infile = open(pathologicaltest_path(Names_Pathological_Test[i]),'r')
    Press = pd.read_csv(infile,sep = '\t',names = ('T','muB','P'))
    press_pathological_test =press_pathological_test.append(pd.DataFrame(Press))
    

for i in range(1,len(Names_Acceptable_Test)):
    infile = open(acceptabletest_path(Names_Acceptable_Test[i]),'r')
    Press = pd.read_csv(infile,sep = '\t',names = ('T','muB','P'))
    press_acceptable_test =press_acceptable_test.append(pd.DataFrame(Press))

#Now we need to reshape the data to matrices

#Define grid
Tgrid = 771
muBgrid = 451
elements = Tgrid*muBgrid
    
#Produce matrices, the classification list contains 0 (pathological) or 1 (acceptable)
#corresponding to the matrix

for i in range(0,int(len(press_acceptable)/347721)):
    data = press_acceptable.iloc[i*elements : (i+1)*elements]['P']
    data = array(data)
    data = data.reshape(Tgrid,muBgrid)
    if i ==0 :
        matpress = [data]
        classification = [1]
    else:
        matpress.append(data)
        classification.append(1)
    
for i in range(0,int(len(press_pathological)/347721)):
    data = press_pathological.iloc[i*elements : (i+1)*elements]['P']
    data = array(data)
    data = data.reshape(Tgrid,muBgrid)
    matpress.append(data)
    classification.append(0)

for i in range(0,int(len(press_acceptable_test)/347721)):
    data = press_acceptable_test.iloc[i*elements : (i+1)*elements]['P']
    data = array(data)
    data = data.reshape(Tgrid,muBgrid)
    if i ==0 :
        matpress_test = [data]
        classification_test = [1]
    else:
        matpress_test.append(data)
        classification_test.append(1)
    
for i in range(0,int(len(press_pathological_test)/347721)):
    data = press_pathological_test.iloc[i*elements : (i+1)*elements]['P']
    data = array(data)
    data = data.reshape(Tgrid,muBgrid)
    matpress_test.append(data)
    classification_test.append(0)
    
### Now we need compute the SVD and extract the coefficients in the muB basis 

#first we want to take the transpose, then we want to center our matrices across the rows, then we compute the SVD

for i in range(0, len(matpress)):
    if i == 0:
        scalled_matpress = [preprocessing.scale(matpress[0].T).T] 
    else:
        scalled_matpress.append(preprocessing.scale(matpress[i].T).T)


for i in range(0, len(matpress_test)):
    if i == 0:
        scalled_matpress_test = [preprocessing.scale(matpress_test[0].T).T] 
    else:
        scalled_matpress_test.append(preprocessing.scale(matpress_test[i].T).T)
        


# Compute SVD and first coefficient 
from matplotlib import pyplot as plt

x_train = []

for j in range(0,len(scalled_matpress)):
    scalled_matpress[j] = pd.DataFrame(scalled_matpress[j])
    scalled_matpress[j].dropna(inplace= True)
    u, s, vh = np.linalg.svd(scalled_matpress[j], full_matrices=True)
    scalled_matpress[j] = array(scalled_matpress[j])
    for i in range(0, scalled_matpress[j].shape[0]-1):
        if i == 0:
            values = [np.dot(vh[:][1],scalled_matpress[j][i][:])/np.dot(vh[:][1],vh[:][1])]
        else:
            values.append(np.dot(vh[:][1],scalled_matpress[j][i][:])/np.dot(vh[:][1],vh[:][1]))
    values = array(values)
    x_train.append(values)

#Same procedure for test data

x_test = []

for j in range(0,len(scalled_matpress_test)):
    scalled_matpress_test[j] = pd.DataFrame(scalled_matpress_test[j])
    scalled_matpress_test[j].dropna(inplace= True)
    u, s, vh = np.linalg.svd(scalled_matpress_test[j], full_matrices=True)
    scalled_matpress_test[j] = array(scalled_matpress_test[j])
    for i in range(0, scalled_matpress_test[j].shape[0]-1):
        if i == 0:
            values = [np.dot(vh[:][1],scalled_matpress_test[j][i][:])/np.dot(vh[:][1],vh[:][1])]
        else:
            values.append(np.dot(vh[:][1],scalled_matpress_test[j][i][:])/np.dot(vh[:][1],vh[:][1]))
    values = array(values)
    x_test.append(values)

    
#Correction to training data missing element

x_train[45] = np.delete(x_train[45], x_train[45][:][768])

#Now we normalize the training and test data for the neural network

x_train = tf.keras.utils.normalize(x_train,axis=0)

x_test = tf.keras.utils.normalize(x_test,axis=0)

classification = array(classification)
classification_test = array(classification_test)

#Our data is ready to be fed to the neural net



model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(770, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(40, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(1, activation = tf.nn.sigmoid))

model.compile(optimizer='adam',
             loss = 'binary_crossentropy',
             metrics=['accuracy'])
model.fit(x_train, classification, epochs = 20)

#save model
model.save('Thermodynamics_Classifier.model')

# Out of sample testing

val_loss, val_acc = model.evaluate(x_test,classification_test)
print('Out of Sample Loss: ')
print(val_loss)
print('Out of Sample Accuracy: ')
print(val_acc)


