# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 17:12:07 2015

@author: amazaspshaumyan
"""
import numpy as np
from scipy.linalg import pinvh

    
def inversion_checker(X,alpha,beta):
    '''
    Checks accuracy of inversion
    '''
    n,m    = X.shape
    #X      = X - np.mean(X,0)
    u,d,vh = np.linalg.svd(X,full_matrices = False)
    dsq    = d**2
    # precision matrix
    S      = beta*np.dot(X.T,X) + alpha*np.eye(m)
    
    # inverting precision : PREVIOUS VERSION
    a1     = np.dot( np.dot(vh.T, np.diag( 1. / (beta*dsq + alpha)) ), vh)
    
    # inverting precision : CURRENT VERSION
    a2     = pinvh(S)
    return [S,a1,a2]
    
    
if __name__ == '__main__':
    X  = np.array([ [ 0.1,  -0.1,  -0.2,   0.02],
                    [ 0.3,  -0.3,  -0.6,   0.06],
                    [ 0.4,  -0.4,  -0.8,   0.08],
                    [ 0.5,  -0.5,  -1.,    0.1 ]])
    # small beta case    
    alpha = 1
    beta = 1000
    print('\n Example 1: beta = {0} \n'.format(beta))
    S,v1, v2 = inversion_checker(X, alpha, beta)
    print("Previous inversion method \n")
    print (v1)
    print("\n Current inversion method \n")
    print (v2)
    
    # large beta case
    beta = 1e+16
    print('\n Example 2: beta = {0}  \n'.format(beta))
    S,v1, v2 = inversion_checker(X, alpha, beta)
    print("Previous inversion method \n")
    print (v1)
    print("\n Current inversion method \n")
    print (v2)
    
    
    X = np.random.random([5,5]) + 0.00000001*np.eye(5)
    #print np.linalg.inv(X)
    #print pinvh(X)
                    
