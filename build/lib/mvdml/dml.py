#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# basic libraries
import numpy as np
import random

# function to create splits for cross-fitting
def partition (list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]

# function to use splits for orthogonalization
def orthog(ind, dep, indices, mod):
    
    # fit the model
    modfit = mod.fit(
        np.delete(ind, indices, 0),
        np.delete(dep, indices, 0).ravel()
    )
    
    # predict
    dephat = modfit.predict(
        ind[indices]
    ).reshape(-1, 1)
    
    # residualize
    depres = dep[indices] - dephat
    
    return depres

def dml(X, y, d, ymod, dmod = None, splits = 2):
    
    # double model logic
    if dmod is None:
        dmod = ymod
    
    # split indices
    I = partition(list(range(len(y))),
                  splits)
    
    # initialize empty lists
    dlist  = np.empty((0,1), float)
    ylist  = np.empty((0,1), float)
    thetas = np.empty((0,1), float)
    
    # perform cross-fitting
    for i in I:
        
        # get orthogonalized treatment
        dorth = orthog(X, d, i, dmod)
        dlist = np.append(dlist, dorth, 0)
        
        # get orthogonalized response
        yorth = orthog(X, y, i, ymod)
        ylist = np.append(ylist, yorth, 0)
        
        # calculate intermediate thetas
        thetas = np.append(thetas,
                           np.mean(dorth*yorth)/np.mean(dorth**2))
    
    # prep pos-orthogonalization regressors
    D = np.hstack( (np.ones((len(dlist), 1)) , dlist) )
    
    # fit the DML2 model
    coefs = np.linalg.lstsq(D, ylist, rcond = None)[0]
    
    # get var-cov matrix for DML2
    res = ylist - (coefs[0]*D[0] + coefs[1]*D[1])
    vcv = np.true_divide(1, len(y) - 2
    )*np.dot(np.dot(res.T,res), np.linalg.inv(np.dot(D.T, D)))
    
    # get DML1 and DML2 coefficient
    theta1 = np.mean(thetas)
    theta2 = coefs[1]
    
    # calculate the dml1 standard error
    se1 = np.sqrt(np.mean( (ylist - theta1*dlist)**2*dlist**2
            ) / (np.mean(dlist**2)**2)
        ) / np.sqrt(len(dlist) - 1)
    
    # calculate the dml2 standard error
    se2 = np.sqrt(np.diagonal(vcv))[1]
    
    # present the output
    return {
        'dml1':{
            'coef_se':np.hstack((theta1, se1))
        },
        'dml2':{
            'coef_se':np.hstack((theta2, se2))
        },
        'orth_data':np.hstack((ylist, dlist)),
        'indices':I
    }