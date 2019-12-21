# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 19:35:30 2019

@author: NaNwani
"""

from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(2)

import numpy as np
import scipy as sp
import matplotlib.pylab as plt


from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense, Input, LeakyReLU, Dropout
from keras.initializers import normal
from keras.utils import to_categorical
from keras.metrics import categorical_accuracy
from keras.losses import categorical_crossentropy
from keras.callbacks import EarlyStopping

# data set parameters
M=20    # number of variables in the data set
N=10     # number of values each variable can take
T=100000 # number of samples

# Multi layer perceptron parameters
num_hidden_nodes = 200
train_epochs = 100
val_split = 0.3

initializer_kernel = normal(mean=0, stddev=2)
initializer_bias = normal(mean=0, stddev=2)


def get_MLP(isTraineeMLP, *numNodesList):
    MLP = Sequential()

    numLayers = len(numNodesList) - 1
    if numLayers == 0:
        print('MLP should have at least one layer')
        return MLP

    if isTraineeMLP:
        MLP.add(Dropout(0.8))

    # hidden layers
    for l in range(numLayers-1):
        if isTraineeMLP:
            MLP.add(Dense(numNodesList[l+1],input_dim=numNodesList[l]))
            #MLP.add(Dropout(0.1))
        else:
            MLP.add(Dense(numNodesList[l+1],
                          input_dim=numNodesList[l],
                          kernel_initializer=initializer_kernel,
                          bias_initializer=initializer_bias))
            MLP.add(LeakyReLU(alpha=0.1))

    # output layer
    if isTraineeMLP:
        MLP.add(Dense(numNodesList[-1],
                  activation='sigmoid',
                  kernel_initializer=initializer_kernel,
                  bias_initializer=initializer_bias))
    else:
        MLP.add(Dense(numNodesList[-1],activation='sigmoid'))

    return MLP


def get_one_hot_mask(mask, N):
    one_hot_mask = np.ones((N,1))*(N*mask)+(np.repeat(range(N),len(mask)).reshape(N,len(mask)))
    one_hot_mask = (one_hot_mask.reshape(N*len(mask))).astype(int)
    return one_hot_mask
    


# function to generate one hot encoded data set: inputs in order: 
# M-number of variables, N-number of states each variable takes, T-number of samples in dataset; C-adjacency matrix of DAG
def generate_dataset_one_hot(M,N,T,C,GTmodels):
    # initialize the independent variable uniformly randomly
    X = np.random.randint(N, size=T)
    
    inputData = np.zeros((T,M))     # raw data from ground truth models
    inputDataOneHot = np.zeros((T,M*N))
    
    inputData[range(T),0] = X
    inputDataOneHot[np.ix_(range(T),range(N))] = to_categorical(X,N)   #one hot encoding
    
    # define the groud truth models and generate the dataset
    for m_index in range(M-1):  #skip first variable which is independent of the rest
        print('ground truth model', m_index+1)
        GTmodels.append(get_MLP(0,M*N,num_hidden_nodes,N))
        
        inp_ind = np.nonzero(C[m_index+1,:])
        mask_ind = np.where(C[m_index+1,:]==0)
        if (len(inp_ind[0]) == 0):
            C[m_index+1,0] = 1
            mask_ind = np.zero(C[m_index+1,:])
        mask_cols = get_one_hot_mask(mask_ind[0], N)
    
        inp = inputDataOneHot.copy()
        inp[np.ix_(range(T),mask_cols)] = 0
    
        pred = GTmodels[m_index].predict(inp)
        #print(pred)
        output = np.argmax(pred,axis=1)
        #print(output)
        inputDataOneHot[range(T),(m_index+1)*N+output] = 1
        inputData[range(T),(m_index+1)] = output
        
    #print(inputData)
    print(sum(inputData))
    
    return inputDataOneHot

# function to define and tain the MLPs
def train_MLPs(M,N,T,C,traineeModels,inpOneHot):
    T1 = 10
    T0 = T - T1
    batch = int(T0 / 100)

    es = EarlyStopping(monitor='loss', min_delta=1e-6, patience=3)
    
    for m_index in range(M):    #create a MLP for each variable
        print('trainee model', m_index)
        
        traineeModels.append(get_MLP(1,M*N,num_hidden_nodes,N))
        traineeModels[m_index].compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
    
        inp = inpOneHot[:T0,:].copy()
        
        op_mask_cols = get_one_hot_mask(np.array([m_index]), N)
        inp[np.ix_(range(T0),op_mask_cols)] = np.zeros((T0,len(op_mask_cols)))
    
        output = inpOneHot[np.ix_(range(T0),op_mask_cols)].copy()
        traineeModels[m_index].fit(inp, output, epochs=train_epochs, batch_size=batch, validation_split=val_split, callbacks=[es])
    
        test_inp = inpOneHot[T0:T0+T1,:].copy()
        true_out = (test_inp[np.ix_(range(T1),op_mask_cols)]).copy()
        test_inp[np.ix_(range(T1),op_mask_cols)] = np.zeros((T1,len(op_mask_cols)))
        
        pred = traineeModels[m_index].predict(test_inp)
        pred = np.argmax(pred,axis=1)
    
        print('test out', pred)
        print('true out', np.argmax(true_out,axis=1))
    

# predict the intervention using trained models
def predict_intervention(M,N,T,intvnDataOneHot,traineeModels):
    nll = np.zeros(M)
    for m_index in range(M):
        mask_cols = get_one_hot_mask(np.array([m_index]), N)
        test_inp = intvnDataOneHot.copy()
        true_out = (test_inp[np.ix_(range(T),mask_cols)]).copy()
        test_inp[np.ix_(range(T2),mask_cols)] = np.zeros((T,len(mask_cols)))
        pred = traineeModels[m_index].predict(test_inp)

        nll[m_index] = np.sum((categorical_crossentropy(true_out, pred))[0])
        
    print(nll)
    pred_intvn = np.argmax(nll)
    
    return pred_intvn
    
# function to generate data after applying soft intervention on intvn_var
def apply_soft_intervention(M,N,T,C,intvnDataOneHot,intvn_var):

    if (intvn_var == 0):
        X = np.random.randint(N, size=T)
        intvnDataOneHot[range(T),X] = 1
        
    for m_index in range(max(intvn_var,1),M):
        mask_ind = np.where(C[m_index,:]==0)
        mask_cols = get_one_hot_mask(mask_ind[0], N)

        inp = intvnDataOneHot.copy()
        inp[np.ix_(range(T),mask_cols)] = 0
        
        if (intvn_var == m_index):    # if intervention node
            intvnModel = get_MLP(0,M*N,num_hidden_nodes,N)
            pred = intvnModel.predict(inp)
        else:
            pred = GTmodels[m_index-1].predict(inp)

        output = np.argmax(pred,axis=1)
        intvnDataOneHot[range(T),m_index*N+output] = 1



if __name__== "__main__":
    # Adjacency matrix definition: using random binary lower triangular matirx
    C = np.random.randint(low = 0, high = 2, size = [M,M]) 
    C = np.tril(C,-1)   # sample only lower traingular portion to make it a DAG
    C[1,0] = 1
    T2 = int(T/2)
    
    GTmodels = []
    inputDataOneHot = generate_dataset_one_hot(M,N,T,C,GTmodels)
    
    traineeModels = []
    train_MLPs(M,N,int(T/2),C,traineeModels,inputDataOneHot[:T2,:])
    
    # derive soft intervention data using a new random MLP for the intervention node
    intvn_var = range(M)
    num_interventions = len(intvn_var)
    I_N_pred = np.zeros(num_interventions)
    
    for intvn in range(num_interventions):
        print('Intervention on model', intvn_var[intvn])
    
        intvnDataOneHot = inputDataOneHot[T2:,:].copy()
        apply_soft_intervention(M,N,T2,C,intvnDataOneHot,intvn_var[intvn])
    
        I_N_pred[intvn] = predict_intervention(M,N,T2,intvnDataOneHot,traineeModels)
    
        print('Predicted intervention on model', I_N_pred[intvn])
    
    print('Actual intervention', intvn_var)
    print('Predicted intervention', I_N_pred)