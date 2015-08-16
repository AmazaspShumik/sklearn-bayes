# -*- coding: utf-8 -*-

import numpy as np

class SparseBayesianLearner(object):
    '''
    Implements Sparse Bayesian Learner, in case no kernel is given this is equivalent
    to regression with automatic relevance determination, if kernel is given it is
    equivalent to relevance vector machine (see Tipping 2001).
    
    Parameters:
    -----------
    
    X: numpy array of size 'n x m'
       Matrix of explanatory variables
       
    Y: numpy vector of size 'n x 1'
       Vector of dependent variables
       
    bias_term: bool
       If True includes bias term in calculation
       
    alpha_max: float
       If alpha corresponding to basis vector will be above alpha_max, then basis
       vector is pruned (i.e. not used in further computations)

    '''
    
    def __init__(self, X, Y, bias_term = False, alpha_max = 1e+9, thresh      = 1e-5,  
                                                                  kernel      = None,
                                                                  kernel_type = None,
                                                                  scaler      = None ):
        # kernelise data if used for RVM        
        if kernel is not None:
            X       = SparseBayesianLearner.kernel_estimation(X,X,kernel_type, scaler)
        
        # dimensionality of data
        self.n, self.m  =  X.shape
        
        # center data
        self.muX        =  np.reshape( np.mean(X, axis = 0), (self.m,1))
        self.muY        =  np.mean(Y)
        self.Y          =  Y - self.muY
        self.X          =  X - self.muX
        
        # convergence parameters
        self.thresh     = thresh
        self.alpha_max  = alpha_max
        
        
    @staticmethod
    def kernel_estimation(K,kernel_type, scaler, p_order, c):
        '''
        Calculates value of kernel for data given in matrix K.
        
        Parameters:
        -----------
        
        K: numpy array of size 'n x m'
           Matrix of explanatory variables
           
        kernel_type: str
           Kernel type , can be 
                                -  gaussian  exp(-(outer_sum(mu,mu') - X*X')/scaler)
                                -  poly      (c + X*X'/ scaler)^p_order
                                
        scaler: float
           value of scaling coefficient 
           
        p_order: float
           Order of polynomial ( valid for polynomial kernel)
           
        c: float
           Constant for polynomial kernel
           
        Returns:
        --------
        
        kernel: numpy array of size 'n x n' - kernel
        '''
        
        def dist(K):
            ''' Calculates distance between observations of matrix K'''
            n,m   = np.shape(K)
            sqrd  = np.reshape( np.sum(K*K, axis = 1), (n,1))
            # outer sum of two m x 1 matrices ( correspond to x^2 + y^2)
            S     = sqrd + sqrd.T
            # outer product of two n x m matrices ( correspond to 2*x*y) 
            I     = 2*np.dot(K,K.T)
            return S-I
                
        if kernel_type == "gaussian":
            distSq = dist(K)
            kernel = np.exp(-distSq/scaler)
            
        elif kernel_type == "poly":
            kernel = (np.dot(X,X.T)/scaler + c)**p_order
            
        return kernel
            
            
            
            
    
            
        
        
           
            
        
        
    
    
    
    
    