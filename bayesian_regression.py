# -*- coding: utf-8 -*-


import numpy as np
import random



class BayesionRegression(object):
    '''
    Bayesian Regression with type maximum likelihood for determining point es
    '''
    
    def __init__(self,X,Y,alpha_beta = None):
        self.X        =  X
        self.Y        =  Y


    def _weights_posterior_params(self,alpha,beta):
        '''
        Calculates parameters of posterior distribution of weights after data was 
        observed.
        
        # Small Theory note:
        ---------------------
        Multiplying likelihood of data on prior of weights we obtain distribution 
        proportional to posterior of weights. By completing square in exponent it 
        is easy to prove that posterior distribution is Gaussian.
        
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
        precision             = beta*np.dot(self.X.T,self.X)+alpha
        
        # use svd decomposition for fast solution
        # posterior_mean = ridge_solution = V*(D/(D**2 + alpha/beta))*U.T*Y
        self.u,self.d,self.v  = np.linalg.svd(X)
        self.diag             = self.d/(self.d**2 + alpha/beta)
        p1                    = np.dot(self.v,np.diag(self.diag))
        p2                    = np.dot(u.T,self.Y)
        w_mu                  = np.dot(p1,p2)
        return [w_mu,precision]
        
        
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
        
        :list of two numpy arrays (each has size 'unknown x 1')
            Parameters of univariate gaussian distribution [mean and variance] 
        
        '''
        mu,precision =  self._coef_posterior(alpha,beta)
        mu_pred      =  np.dot(x,mu)
        S            =  np.dot(np.dot(self.v.T,np.diag(self.diag)),self.v) / beta
        var          =  np.array([1.0/beta + np.dot(u,S.dot(u)) for u in x])
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
    
    