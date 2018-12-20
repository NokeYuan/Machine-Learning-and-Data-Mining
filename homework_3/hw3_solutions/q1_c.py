
#Example code for Q1 c)

import numpy as np  

  
def grads(X, t, w, b):

    y = np.dot(X,w)+b  
    dif = y-t
    H_prime = np.where(np.abs(dif) <= delta, dif, delta*np.sign(dif))

    dw = np.dot(X.T, H_prime) / X.shape[0]
    db = np.mean(H_prime,0)

    return dw, db




N = 10  # number of datapoints
D = 5   # dimensionality of datapoints

X = np.zeros((N,D))  # placeholder dataset
t = np.zeros((N,1))

delta = 1. #Huber loss hyperparameter

w = np.zeros((D,1)) 
b = 0  
 
lr = 0.01  # learning rate  
for i in range (0,1000):  
    dw, db = grads(X, t, w, b)

    w = w - lr*dw
    b = b - lr*db









