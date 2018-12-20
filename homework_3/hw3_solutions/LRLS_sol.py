
# Example code for Q2

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

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

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(1,d),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()
    
    return losses
 
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
    dist = l2(test_datum,x_train)
    dist = dist-np.min(dist,axis=1)
    alpha = np.exp(-dist/(2*tau**2))
    alpha = (alpha/alpha.sum()).flatten()
    assert(np.abs(alpha.sum()-1)<0.0001)
    A = np.diag(alpha)
    w =  np.linalg.solve(x_train.transpose().dot(A).dot(x_train)+lam*np.eye(d), (x_train.transpose().dot(A)).dot(y_train))
    return w.dot(test_datum.flatten())




def run_k_fold(X,Y,taus,k):
    '''
    Input: X is the N x d design matrix
           Y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## cheating
    kf = KFold(n_splits=k)
    kf.get_n_splits(X)
    kf_losses = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        kf_losses.append(run_on_fold(X_test, y_test, X_train, y_train, taus).reshape(-1,1))
    return np.concatenate(kf_losses,axis=1).mean(axis=1)

def run_k_fold_base(X,Y,taus,k):
    '''
    Input: X is the N x d design matrix
           Y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## cheating
    kf = KFold(n_splits=k)
    kf.get_n_splits(X)
    kf_losses = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        kf_losses.append(((y_test-pred)**2).mean())
    return np.array(kf_losses)


# In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
taus = np.logspace(1.0,3,200)
losses = run_k_fold(x,y,taus,k=10)
plt.plot(losses)
print("min loss = {}".format(losses.min()))

