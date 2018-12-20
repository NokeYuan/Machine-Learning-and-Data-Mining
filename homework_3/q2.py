# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.special import logsumexp
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    ## TODO
    #compute a(i)
    #reshape test_datum , so it can fit the dimension with l2 function.
    test_datum = test_datum.reshape(1, test_datum.shape[0])
    distance = l2(test_datum,x_train)
    a_numerator = np.exp((-distance)/(2*(tau**2)))
    a_denominator = np.exp(logsumexp((-distance)/(2*(tau**2))))
    a_i = np.diagflat(a_numerator/a_denominator)

    #compute coef matrix and dependent variables of W*
    coef_matrix = x_train.T.dot(a_i).dot(x_train) + lam*np.identity(x_train.shape[1])
    dependent_variables =x_train.T.dot(a_i).dot(y_train)

    #find w
    w = np.linalg.solve(coef_matrix,dependent_variables)
    y_hat = test_datum.dot(w)

    return y_hat



def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''


    #  Shuffle x and y set according to val_frac
    frac_index = round(x.shape[0] * val_frac)
    validation_index = idx[0:frac_index]
    training_index = idx[frac_index:]

    x_validation = x[validation_index]
    x_traning = x[training_index]

    y_validation = y[validation_index]
    y_traning = y[training_index]

    training_losses = []
    validation_losses = []

    #for each taus, find its average loss and append it to trainning_losses
    for i in range(len(taus)):
        training_loss = 0
        for j in range(len(x_traning)):
            prediction = LRLS(x_traning[j],x_traning,y_traning,taus[i])
            each_loss = 0.5*((prediction - y_traning[j])**2)
            training_loss += each_loss
        average_loss = training_loss/len(x_traning)
        training_losses.append(average_loss)

    # for each taus, find its average loss and append it to validation_losses
    for i in range(len(taus)):
        validation_loss = 0
        for j in range(len(x_validation)):
            prediction = LRLS(x_validation[j],x_traning,y_traning,taus[i])
            each_loss = 0.5*((prediction - y_validation[j])**2)
            validation_loss += each_loss
        average_loss = validation_loss/len(x_validation)
        validation_losses.append(average_loss)

    #vectorize result
    return np.asarray(training_losses),np.asarray(validation_losses)



if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(train_losses)
    plt.semilogx(test_losses)
    plt.show()

