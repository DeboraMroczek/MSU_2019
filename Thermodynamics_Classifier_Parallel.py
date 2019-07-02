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
from itertools import chain
import math
import multiprocessing as mp
import datatable as dt
from tensorflow.keras.backend import set_session

os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Define path for data set
DATA_ID = "muBC350_01grid"


#Define T-muB grid size

Tsize = 771
muBsize = 451


#Define functions to read files contained in training set
def data_path(filename):
    return os.path.join(DATA_ID, filename)

# Define Function that imports data
# def get_data(Name):
#     parts = Name.split('_')

#     infile = open(data_path(Name),'r')
#     Press = pd.read_csv(infile,sep = '\t',names = ('T','muB','P')) 


#     return Press

# def get_w(Name):
#     parts = Name.split('_')
#     w = float("{0:.4f}".format(float(parts[7])/100.0))

#     return w

# def get_w(Name):
#     parts = Name.split('_')
#     rho = float("{0:.4f}".format(float(parts[8])/100.0))

#     return rho

#Name of the folder for training sets
# Names_Data = os.listdir("01_grid") - Can be used but doesn't prevent hidden files such as .DS_store from being imported which WILL crash the code

#this format prevents that from happening by ensuring the file has the correct format
Names_Data = []

for item in os.listdir(DATA_ID):
    if not item.startswith('.') and os.path.isfile(os.path.join(DATA_ID, item)):
        Names_Data.append(item)

press_data = []
w = []
rho = []
matpress = []

#Define grid
Tgrid = Tsize
muBgrid = muBsize
elements = Tgrid*muBgrid

for filename in Names_Data:
    parts = filename.split('_')
    data = dt.fread(data_path(filename))
    data = array(data[:,'C2'])
    data = data.reshape(Tgrid,muBgrid)
    matpress.append(data)
    del data
    w.append(float("{0:.4f}".format(float(parts[7])/100.0)))
    rho.append(float("{0:.4f}".format(float(parts[8])/100.0)))
    print('Reading files ', len(w), '/',len(Names_Data), end="\r")


print('* Matrices created successfully - Scalling data *')   

    
### Now we need compute the SVD and extract the coefficients in the muB basis 

#first we want to take the transpose, then we want to center our matrices across the rows, then we compute the SVD

scalled_matpress = []
for data in matpress:
    scalled_matpress.append(preprocessing.scale(data.T).T)
    print (len(scalled_matpress),'/',len(matpress), end="\r")
      
del matpress

# Compute SVD and first coefficient 

print('* Computing PCA coefficients *')

x_eval = []

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
    x_eval.append(values)
    print ('This might take a moment ...', end="\r")


print('* PCA computed successfully *')

del scalled_matpress    

#Now we normalize the training and test data for the neural network
print('* Normalizing data and calling the model *')
x_eval = tf.keras.utils.normalize(x_eval,axis=0)

#Call the model 
classifier = tf.keras.models.load_model('Thermodynamics_Classifier.model')
predictions = classifier.predict(x_eval)


with open('classification_raw.dat','w') as f:
    for i in range(0,len(predictions)):
        f.write("%f %f %f \n" % (w[i],rho[i],predictions[i][0]))
f.close()


list_acceptable_w = []
list_acceptable_rho = []
list_pathological_w = []
list_pathological_rho = []


for i in range(0,len(x_eval)):
    if abs(predictions[i][0]) < abs(predictions[i][0] - 1):
        list_pathological_w.append([w[i]])
        list_pathological_rho.append([rho[i]])
    else:
        list_acceptable_w.append([w[i]])
        list_acceptable_rho.append([rho[i]])


with open('predictions.dat','w') as f:
    for i in range(0,len(predictions)):
        f.write("%f %f %f %f\n" % (w[i], rho[i], predictions[i][0], predictions[i][1]))
f.close()

flattened_pathological_w = [val for sublist in list_pathological_w for val in sublist]
flattened_pathological_rho = [val for sublist in list_pathological_rho for val in sublist]
flattened_acceptable_w = [val for sublist in list_acceptable_w for val in sublist]
flattened_acceptable_rho = [val for sublist in list_acceptable_rho for val in sublist]

with open('pathological_w.dat', 'w') as f:
    for item in flattened_pathological_w:
        f.write("%f\n" % item)

f.close()

with open('pathological_rho.dat', 'w') as f:
    for item in flattened_pathological_rho:
        f.write("%f\n" % item)

f.close()

with open('acceptable_w.dat', 'w') as f:
    for item in flattened_acceptable_w:
        f.write("%f\n" % item)

f.close()

with open('acceptable_rho.dat', 'w') as f:
    for item in flattened_acceptable_rho:
        f.write("%f\n" % item)

f.close()








