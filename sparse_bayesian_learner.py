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
                                                                  scaler      = None,
                                                                  method      = "EM",
                                                                  max_iter    = 100):
        # kernelise data if used for RVM        
        if kernel is not None:
            X                = SparseBayesianLearner.kernel_estimation(X,X,kernel_type, scaler)
        
        # dimensionality of data
        self.n, self.m       =  X.shape
        
        # center data
        self.muX             =  np.reshape( np.mean(X, axis = 0), (self.m,1))
        self.muY             =  np.mean(Y)
        self.Y               =  Y - self.muY
        self.X               =  X - self.muX
        
        # convergence parameters & maximum allowed number of iterations
        self.thresh          = thresh
        self.alpha_max       = alpha_max
        self.max_iter        = max_iter
        
        # method for evidence approximation , either "EM" or "fixed-point"
        self.method          = method
        
        
    def fit(self):
        '''
        Fits Sparse Bayesian Learning Algorithm 
        '''
        
        # initialise precision parameters for prior & likelihood
        diagA    = np.random.random(self.m)
        beta     = np.random.random()
        # array for easy broadcasting ( using diagonal matrix is very expensive)
        d        = np.zeros([self.m,1])
        
        for i in range(self.max_iter):
            
            # set of features that will be used in computation
            active = diagA < alpha_max
            X      = self.X[:,active]
            self.m = np.sum(active)
            alpha  = diagA[active]
            
            # calculate posterior mean & precision of weights ( with EM method 
            # for evidence approximation this corresponds to E-step )
            
            # precision parameter
            S            = beta*np.dot(X.T,X)
            np.fill_diagonal(S, alpha)
            
            # instead of inversion of precision matrix we use Cholesky decomposition
            # to find mu
            m              = np.dot(X.T,self.Y)*beta
            L              = np.linalg.cholesky(S)
            Z              = np.linalg.solve(L,m)
            Mu             = np.linalg.solve(L.T,Z)
            
            # error term
            err            = self.Y - np.dot(X,mu)
            err_sq         = np.dot(err,err)
            
            if self.method == "fixed-point":
                
                # update precision parameters of likelihood & prior
                gamma   = 1 - diagA / (beta*lambda_sq + diagA)
                diagA   = gamma/mu**2
                beta    = (self.n - np.sum(gamma))/err_sq
               
            elif self.method == "EM":
                
                # M-step , finds new A and beta which maximize log likelihood
                beta = self.n / (err_sq + )
                
                
            
                
            
                
                
            
            
        
        
        
        
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
        # inner function for distance calculation
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
            kernel = (np.dot(K,K.T)/scaler + c)**p_order
            
        return kernel
            
            
            
            
    
            
        
        
           
            
        
        
    
    
    
    
    