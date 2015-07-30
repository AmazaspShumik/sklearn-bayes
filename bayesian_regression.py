# -*- coding: utf-8 -*-


import numpy as np
import random

class BayesionRegression(object):
    
    def __init__(self,X,Y,alpha_beta = None):
        self.X        =  X
        self.Y        =  Y

        
        
    def _weights_posterior_params(self,alpha,beta):
        '''
        Calculates parameters of posterior distribution of weights. Since it is 
        assumed prior is Gaussian with precision alpha then posterior of weights
        is also Gaussian
        
        Parameters:
        -----------
        
        alpha: float
            Precision parameter for prior distribution of weights
            
        beta: float
            Precision parameter for noise distribution
            
        Returns:
        --------
        
        : list of two numpy arrays
           First element of list is mean and second is precision of multivariate 
           Gaussian distribution
           
        '''
        X_t       = self.X.T
        precision = beta*np.dot(X_t,self.X)+alpha
        mu        = beta*np.linalg.solve(precision,np.dot(X_t,self.Y))
        return [mu,precision]
        
        
    def _pred_dist_params(self,alpha,beta,x):
        '''
        Calculates parameters of predictive distribution
        
        Parameters:
        -----------
        
        alpha: float
            Precision parameter for prior distribution of weights
            
        beta: float
            Precision parameter for noise distribution
            
        x: numpy array of size 'unknown x m'
            Set of features for which corresponding responses should be predicted
            
        Returns:
        ---------
        
        :list of two numpy arrays
            Parameters of univariate gaussian distribution [mean and variance]
        
        '''
        mu,precision =  self._coef_posterior(alpha,beta)
        
        # precision is Positiove Definite (due to + alpha*I term), so there
        # is no numerical issue inverting precision matrix even in presence
        # of multicollinearity in design matrix
        S            =  np.linalg.inv(precision)
        mu_pred      =  np.dot(x,mu)
        var          =  lambda v: np.dot(np.dot(S,v),v)
        sigma_pred   =  1.0/beta + np.sum([var(u) for u in x])
        return [mu_pred,sigma_pred]
        
        
    def _em_evidence_approx(self, max_iter = 20, convergence_thresh, init_params):
        '''
        Performs evidence approximation using EM algorithm to find parameters
        alpha and beta in case they are not given. Maximizes type II maximum
        likelihood
        
        Parameters:
        -----------
        
        max_iter: int
            Maximum number of iteration for EM algorithm
            
        
        
        
        '''
        
        # number of observations and number of paramters
        n,m   = np.shape(self.X)
        
        # initial values of alpha and beta ( can be considered as initial E-step)
        alpha, beta = np.random.random(2)
        
        for i in range(max_iter):
            
            # M-step, find mean vector and precision matrix for posterior 
            # distribution of weights    
            mu, L = self._weights_posterior_params(alpha,beta)
             
            # E-step, find paramters alpha and beta that determine posterior 
            # distribution of weights (given mean vector and precision matrix)
            alpha = float(m)/()
            beta  = 
        
        
    def predict_prob(self):
        pass
    
    def predict_mean(self):
        pass
    
    