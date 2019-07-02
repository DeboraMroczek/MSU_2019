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
from sklearn.model_selection import train_test_split


#Define path for training sets
PATHOLOGICAL_ID = "Pathological-Training/"
ACCEPTABLE_ID = "Non-Pathological-Training"


#Define functions to read files contained in training set
def pathological_path(filename):
    return os.path.join(PATHOLOGICAL_ID, filename)

def acceptable_path(filename):
    return os.path.join(ACCEPTABLE_ID, filename)

def get_w(dTC,TC):
    w = dTC/TC 
    return w
 
def get_rho(dTC,dmuBC):
    rho = dmuBC/dTC  
    return rho

#Name of the folder for training sets
Names_Pathological = os.listdir('Pathological-Training')
Names_Acceptable = os.listdir('Non-Pathological-Training')


#Open first file and define pandas data frame for the pathological portion of the training set 
infile = open(pathological_path(Names_Pathological[0]),'r')
press_pathological = pd.read_csv(infile,sep = '\t',names = ('T','muB','P'))

infile = open(acceptable_path(Names_Acceptable[0]),'r')
press_acceptable = pd.read_csv(infile,sep = '\t',names = ('T','muB','P'))


#Import training files and define Pandas data frame

for i in range(1,len(Names_Pathological)):
    infile = open(pathological_path(Names_Pathological[i]),'r')
    Press = pd.read_csv(infile,sep = '\t',names = ('T','muB','P'))
    press_pathological =press_pathological.append(pd.DataFrame(Press))
    

for i in range(1,len(Names_Acceptable)):
    infile = open(acceptable_path(Names_Acceptable[i]),'r')
    Press = pd.read_csv(infile,sep = '\t',names = ('T','muB','P'))
    press_acceptable =press_acceptable.append(pd.DataFrame(Press))


with open('rho_w_pathological.dat','w') as f:
    for filename in Names_Pathological:
        parts = filename.split('_')
        w = get_w(float(parts[7]),float(parts[3]))
        rho = get_rho(float(parts[7]), float(parts[8]))
        f.write("%f %f\n" % (w, rho))

with open('rho_w_acceptable.dat','w') as f:
    for filename in Names_Acceptable:
        parts = filename.split('_')
        w = get_w(float(parts[7]),float(parts[3]))
        rho = get_rho(float(parts[7]), float(parts[8]))
        f.write("%f %f\n" % (w, rho))


















# #Now we need to reshape the data to matrices

# #Define grid
# Tgrid = 771
# muBgrid = 451
# elements = Tgrid*muBgrid
    
# #Produce matrices, the classification list contains 0 (pathological) or 1 (acceptable)
# #corresponding to the matrix (very non-python way of doing this)

# for i in range(0,int(len(press_acceptable)/347721)):
#     data = press_acceptable.iloc[i*elements : (i+1)*elements]['P']
#     data = array(data)
#     data = data.reshape(Tgrid,muBgrid)
#     if i ==0 :
#         matpress = [data]
#         classification = [[0,1]]
#     else:
#         matpress.append(data)
#         classification.append([0,1])
    
# for i in range(0,int(len(press_pathological)/347721)):
#     data = press_pathological.iloc[i*elements : (i+1)*elements]['P']
#     data = array(data)
#     data = data.reshape(Tgrid,muBgrid)
#     matpress.append(data)
#     classification.append([1,0])

    
# ### Now we need compute the SVD and extract the coefficients in the muB basis 

# #first we want to take the transpose, then we want to center our matrices across the rows, then we compute the SVD

# for i in range(0, len(matpress)):
#     if i == 0:
#         scalled_matpress = [preprocessing.scale(matpress[0].T).T] 
#     else:
#         scalled_matpress.append(preprocessing.scale(matpress[i].T).T)


# # Compute SVD and first coefficient 
# from matplotlib import pyplot as plt

# processed_data = []

# for j in range(0,len(scalled_matpress)):
#     scalled_matpress[j] = pd.DataFrame(scalled_matpress[j])
#     scalled_matpress[j].dropna(inplace= True)
#     u, s, vh = np.linalg.svd(scalled_matpress[j], full_matrices=True)
#     scalled_matpress[j] = array(scalled_matpress[j])
#     for i in range(0, scalled_matpress[j].shape[0]-1):
#         if i == 0:
#             values = [np.dot(vh[:][1],scalled_matpress[j][i][:])/np.dot(vh[:][1],vh[:][1])]
#         else:
#             values.append(np.dot(vh[:][1],scalled_matpress[j][i][:])/np.dot(vh[:][1],vh[:][1]))
#     values = array(values)
#     processed_data.append(values)
    
# #Correction to training data missing element

# # processed_data[45] = np.delete(processed_data[45], processed_data[45][:][768])

# #Turn the classification list into an array 
# classification = array(classification)

# #Now we split the data between training and testing

# train_size = 0.8
# test_size = 1 - train_size
# X_train, X_test, Y_train, Y_test = train_test_split(processed_data, classification, train_size=train_size,
#                                                     test_size=test_size)

# #Now we normalize the training and test data for the neural network

# X_train = tf.keras.utils.normalize(X_train,axis=0)

# X_test = tf.keras.utils.normalize(X_test,axis=0)

# #Our data is ready to be fed to the neural net

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(80, activation = tf.nn.relu))
# model.add(tf.keras.layers.Dense(80, activation = tf.nn.relu))
# model.add(tf.keras.layers.Dense(2, activation = tf.nn.softmax))

# model.compile(optimizer='adam',
#              loss = 'binary_crossentropy',
#              metrics=['accuracy'])
# model.fit(X_train, Y_train, epochs = 20)

# #save model
# model.save('Thermodynamics_Classifier.model',include_optimizer=False)

# # Out of sample testing

# val_loss, val_acc = model.evaluate(X_test,Y_test)
# print('Out of Sample Loss: ')
# print(val_loss)
# print('Out of Sample Accuracy: ')
# print(val_acc)


