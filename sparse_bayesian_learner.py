# -*- coding: utf-8 -*-

import numpy as np


class SparseBayesianLearner(object):
    '''
    Implements Sparse Bayesian Learner, in case no kernel is given this is equivalent
    to regression with automatic relevance determination, if kernel is given it is
    equivalent to relevance vector machine (see Tipping 2001).
    
    Parameters:
    -----------
    


    alpha_max: float
       If alpha corresponding to basis vector will be above alpha_max, then basis
       vector is pruned (i.e. not used in further computations)
       
    thresh: float
       Convergence parameter
       
    kernel: str or None
       Type of kernel to use [currently possible: None, 'gaussian', 'poly']
       
    scaler: float (used for kernels)
       Scaling parameters for kernels
       
    method: str
       Method to fit evidence approximation, currently available: "EM","fixed-point"
       ( Note in 2009 Tipping proposed much faster algorithm for fitting RVM , it is 
       not implemented in this version of code)
       
    max_iter: int
       Maximum number of iterations
       
    verbose: str
       If True prints messages about progress in computation
       
       
    References:
    -----------
    Tipping 2001, Sparse Bayesian Learning and Relevance Vector Machine
    Bishop 2006, Pattern Recognition and Machine Learning (Chapters 3,6,7,9)

    '''
    
    def __init__(self, alpha_max = 1e+9, thresh      = 1e-30, kernel      = None,
                                                              scaler      = None,
                                                              method      = "fixed-point",
                                                              max_iter    = 1500,
                                                              p_order     = 2,
                                                              verbose     = False):
        self.verbose         = verbose
        self.kernel          = kernel
        self.scaler          = scaler 
        # if polynomial kernel, add order
        self.p_order         = p_order
        
        # convergence parameters & maximum allowed number of iterations
        self.thresh          = thresh
        self.alpha_max       = alpha_max
        self.max_iter        = max_iter
        
        # method for evidence approximation , either "EM" or "fixed-point"
        self.method          = method
        
        # parameters computed while fitting model
        self.Mu              = 0
        self.Sigma           = 0
        self.active          = 0
        self.diagA           = 0
        self.gamma           = 0
        self.support_vecs    = None
        
        
        
    def fit(self,X,Y):
        '''
        Fits Sparse Bayesian Learning Algorithm, writes mean and covariance of
        posterior distribution to instance variables. 
        
        Parameters:
        -----------        
        
        X: numpy array of size 'n x m'
           Matrix of explanatory variables
       
        Y: numpy vector of size 'n x 1'
           Vector of dependent variables
           
        '''
        # kernelise data if used for RVM     
        if self.kernel is not None:
           Xraw              = X  
           X                 = SparseBayesianLearner.kernel_estimation(X,X,self.kernel, self.scaler, self.p_order)
           
        # dimensionality of data
        self.n, self.m       =  X.shape
        
        # center data
        self.muX             =  np.reshape( np.mean(X, axis = 0), (self.m,1) )
        self.muY             =  np.mean(Y)
        Yc                   =  Y - self.muY
        Xc                   =  X - self.muX.T
        
        # initialise precision parameters for prior & likelihood
        diagA    = np.random.random(self.m)
        beta     = np.random.random()
        
        for i in range(self.max_iter):
            
            # set of features that will be used in computation
            active            = diagA < self.alpha_max
            self.m            = np.sum(active)
            if self.m == 0:
                raise ValueError("All features were pruned. Check value for parameter alpha_max")
                        
            # calculate posterior mean & covariance of weights ( with EM method 
            # for evidence approximation this corresponds to E-step )
            Mu,Sigma          =  self._posterior_params(Xc[:,active],Yc,diagA[active],beta)
            
            # error term
            err               = Yc - np.dot(Xc[:,active],Mu)
            err_sq            = np.dot(err,err)
            gamma             = 1 - diagA[active]*np.diag(Sigma)
            
            # save previous values of alphas and beta
            old_A             = diagA
            old_beta          = beta
            
            if self.method == "fixed-point":
                # update precision parameters of likelihood & prior
                diagA[active] = gamma/Mu**2
                beta          = (self.n - np.sum(gamma))/err_sq
                
            elif self.method == "EM":
                # M-step , finds new A and beta which maximize log likelihood
                diagA[active] = 1.0 / (Mu**2 + np.diag(Sigma))
                beta          = self.n /(err_sq + np.sum(gamma)/beta)
                
            # if change in alpha & beta is below threshold then terminate 
            # iterations
            delta_alpha = np.max(abs(old_A[active] - diagA[active]))
            delta_beta  = abs(old_beta - beta)
            if self.verbose:
                print "iteration {0} is completed ".format(i)
            if  delta_alpha < self.thresh and delta_beta < self.thresh:
                if self.verbose:
                   print 'evidence approximation algorithm terminated...'
                break

            
        self.active          = diagA < self.alpha_max
        self.diagA           = diagA
        self.beta            = beta
        if self.kernel is not None:
            self.support_vecs    = Xraw[self.active,:]
        self.m               = np.sum(self.active)
        # posterior mean and covariance after last update of alpha & beta
        self.Mu,self.Sigma   = self._posterior_params(Xc[:, self.active],Yc,diagA[self.active],beta)
        
        
        
        
    def predict(self,x):
        '''
        Calculates parameters of predictive distribution, returns mean and standard 
        deviation of prediction.
        
        Parameters:
        -----------
        
        x: numpy array of size 'unknown x m'
           Matrix of test explanatory variables.
           
        Returns:
        --------
        
        [mu,var]:
                 mu: numpy array of size 'unknown x 1'
                     vector of means for each data point
                 std: numpy array of size 'unknown x 1'
                     vector of variances for each data point
        '''
        # kernelise data if required and choose relevant features ( support vectors )
        if self.kernel is not None:
            x = SparseBayesianLearner.kernel_estimation(x,self.support_vecs,
                                                          self.kernel,
                                                          self.scaler,
                                                          self.p_order)
        else:
            x = x[:,self.active]
            
        # center data to account for bias term
        x    =  x - self.muX[self.active,:].T
        
        # mean of predictive distribution
        mu   =  np.dot(x,self.Mu) + self.muY
        var  =  1.0 / self.beta + np.dot( np.dot( x , self.Sigma ), x.T )
        return [mu,var]
        
        
        
    def _posterior_params(self,X,Y,diagA,beta):
        '''
        Calculates mean and covariance of posterior distribution of weights.
        
        Parameters:
        -----------
        
        X: numpy array of size 'n x m (active)'
           Matrix of active explanatory features
           
        Y: numpy array of size 'n x 1'
           Vector of explanatory variables
        
        diagA: numpy array of size 'm x 1'
           Vector of diagonal elements for precision of prior
           
        beta: float
           Precision parameter of likelihood
           
        Returns:
        --------
        [Mu,Sigma] : 
                   Mu: numpy array of size 'm x 1', mean of posterior
                   Sigma: numpy array of size 'm x m', covariance matrix of posterior
        
        '''
        # precision parameter of posterior
        S                 = beta*np.dot(X.T,X)
        np.fill_diagonal(S, np.diag(S) + diagA)
        # calculate mean & covariance from precision using svd
        u,d,v             = np.linalg.svd(S, full_matrices = False)
        inv_eigvals       = 1.0 / d
        Sigma             = np.dot(v.T,v*np.reshape(inv_eigvals,(self.m,1)))
        Mu                = beta*np.dot(Sigma, np.dot(X.T,Y))
        return [Mu,Sigma]
        
    
    
    @staticmethod
    def kernel_estimation(K1,K2,kernel_type, scaler, p_order):
        '''
        Calculates value of kernel for data given in matrix K.
        
        Parameters:
        -----------
        
        K1: numpy array of size 'n1 x m'
           Matrix of variables
           
        K2: numpy array of size 'n2 x m'
           Matrix of variables (in case of prediction support vectors)
           
        kernel_type: str
           Kernel type , can be 
                                -  gaussian  exp(-(outer_sum(mu,mu') - X*X')/scaler)
                                -  poly      (c + X*X'/ scaler)^p_order
                                
        scaler: float
           value of scaling coefficient 
           
        p_order: float
           Order of polynomial ( valid for polynomial kernel)
           
        Returns:
        --------
        
        kernel: numpy array of size 'n x n' - kernel
        '''
        # inner function for distance calculation
        def dist(K1,K2):
            ''' Calculates distance between observations of matrix K'''
            n1,m1 = K1.shape
            n2,m2 = K2.shape
            # outer sum of two m x 1 matrices ( correspond to x^2 + y^2)
            K1sq  = np.outer( np.sum(K1**2, axis = 1), np.ones(n2) )
            K2sq  = np.outer( np.sum(K2**2, axis = 1), np.ones(n1) ).T
            #  correspond to 2*x*y
            K12   = 2*np.dot(K1,K2.T)
            return K1sq - K12 + K2sq 
                
        if kernel_type == "gaussian":
            distSq = dist(K1,K2)
            kernel = np.exp(-distSq/scaler)
            
        elif kernel_type == "poly":
            kernel = (np.dot(K1,K2.T)/scaler + 1)**p_order
            
        return kernel
