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

#Define path for data set
DATA_ID = "01_grid"


#Define T-muB grid size

Tsize = 771
muBsize = 451


#Define functions to read files contained in training set
def data_path(filename):
    return os.path.join(DATA_ID, filename)

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

#Name of the folder for training sets
# Names_Data = os.listdir("01_grid") - Can be used but doesn't prevent hidden files such as .DS_store from being imported which WILL crash the code

#this format prevents that from happening by ensuring the file has the correct format
Names_Data = []

for item in os.listdir(DATA_ID):
    if not item.startswith('.') and os.path.isfile(os.path.join(DATA_ID, item)):
        Names_Data.append(item)


Name_Elements = []
for i in range (0,len(Names_Data)):
    Name_Elements.append(Names_Data[i].split('_'))

w = []
rho = []

for i in range(0, len(Name_Elements)):
    w.append([float("{0:.4f}".format(float(Name_Elements[i][7])/float(Name_Elements[i][4])))])
    rho.append([float("{0:.4f}".format(float(Name_Elements[i][8])/float(Name_Elements[i][7])))]) 

#Open first file and define pandas data frame 
infile = open(data_path(Names_Data[0]),'r')
press_data = pd.read_csv(infile,sep = '\t',names = ('T','muB','P'))


print('* Importing data files from ',DATA_ID, ' *')

#Import remaining files and define Pandas data frame
for i in range(1, len(Names_Data)):
    infile = open(data_path(Names_Data[i]),'r')
    Press = pd.read_csv(infile,sep = '\t',names = ('T','muB','P'))
    print (i,'/',len(Names_Data), end="\r")
    press_data = press_data.append(pd.DataFrame(Press))


print('* Data stored successfully - Generating T-muB grid *')


#Now we need to reshape the data to matrices

#Define grid
Tgrid = Tsize
muBgrid = muBsize
elements = Tgrid*muBgrid
    
#Produce matrices

for i in range(0,int(len(press_data)/347721)):
    data = press_data.iloc[i*elements : (i+1)*elements]['P']
    data = array(data)
    data = data.reshape(Tgrid,muBgrid)
    if i == 0:
        matpress = [data]
    else:
        matpress.append(data)
    print (i,'/',len(Names_Data), end="\r")
    
del press_data

print('* Matrices created successfully - Scalling data *')   

    
### Now we need compute the SVD and extract the coefficients in the muB basis 

#first we want to take the transpose, then we want to center our matrices across the rows, then we compute the SVD

for i in range(0, len(matpress)):
    if i == 0:
        scalled_matpress = [preprocessing.scale(matpress[0].T).T] 
    else:
        scalled_matpress.append(preprocessing.scale(matpress[i].T).T)
    print (i,'/',len(matpress), end="\r")
      
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

list_acceptable_w = []
list_acceptable_rho = []
list_pathological_w = []
list_pathological_rho = []


for i in range(0,len(x_eval)):
    if abs(predictions[i][0] - 0) < abs(predictions[i][0] - 1):
        list_pathological_w.append(w[i])
        list_pathological_rho.append(rho[i])
    else:
        list_acceptable_w.append(w[i])
        list_acceptable_rho.append(rho[i])


flattened_pathological_w = [val for sublist in list_pathological_w for val in sublist]
flattened_pathological_rho = [val for sublist in list_pathological_rho for val in sublist]
flattened_acceptable_w = [val for sublist in list_acceptable_w for val in sublist]
flattened_acceptable_rho = [val for sublist in list_acceptable_rho for val in sublist]

with open('pathological_w.dat', 'w') as f:
    for item in flattened_pathological_w:
        val = round_half_up(item,1)
        if val == 0.0:
            val = 0.1
        f.write("%f\n" % val)

f.close()

with open('pathological_rho.dat', 'w') as f:
    for item in flattened_pathological_rho:
        val = round_half_up(item,1)
        if val == 0.0:
            val = 0.1
        f.write("%f\n" % val)

f.close()

with open('acceptable_w.dat', 'w') as f:
    for item in flattened_acceptable_w:
        val = round_half_up(item,1)
        if val == 0.0:
            val = 0.1
        f.write("%f\n" % val)

f.close()

with open('acceptable_rho.dat', 'w') as f:
    for item in flattened_acceptable_rho:
        val = round_half_up(item,1)
        if val == 0.0:
            val = 0.1
        f.write("%f\n" % val)

f.close()














