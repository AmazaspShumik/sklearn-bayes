# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import multivariate_normal as mvn


class BayesionRegression(object):
    '''
    Bayesian Regression with type maximum likelihood for determining point estimates
    for precision variables alpha and beta, where alpha is precision of prior of weights
    and beta is precision of likelihood
    
    Parameters:
    -----------
    
    X: numpy array of size 'n x m'
       Matrix of explanatory variables (should not include bias term)
       
    Y: numpy arra of size 'n x 1'
       Vector of dependent variables (we assume)
       
    thresh: float
       Threshold for convergence for alpha (precision of prior)
       
    '''
    
    def __init__(self,X,Y, thresh = 1e-5):
        
        # center input data for simplicity of further computations
        self.X                 =  self._center(X)
        self.Y                 =  self._center(Y)
        self.thresh            =  thresh
        
        # to speed all further computations save svd decomposition and reuse it later
        self.u,self.d,self.v   =  np.linalg.svd(self.X)
        
        
    @staticmethod
    def _center(X):
        '''
        Centers data by removing mean, this simplifies further computation
        '''
        if len(X.shape) > 1:
            return X - np.mean(X, axis = 0)
        else:
            return X - np.mean(X)



    def _weights_posterior_params(self,alpha,beta):
        '''
        Calculates parameters of posterior distribution of weights after data was 
        observed. Uses svd for fast calculaltions.
        
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
        self.diag             = self.d/(self.d**2 + alpha/beta)
        p1                    = np.dot(self.v.T,np.diag(self.diag))
        p2                    = np.dot(self.u.T,self.Y)
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
        S            =  np.dot(np.dot(self.v,np.diag(self.diag)),self.v.T) / beta
        var          =  np.array([1.0/beta + np.dot(u,S.dot(u)) for u in x])
        return [mu_pred,var]
        
        
    def _em_evidence_approx(self, max_iter = 20):
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
            pass
            
            
    def _fixed_point_evidence_approx(self,max_iter):
        '''
        Performs evidence approximation using fixed point algorithm, empirical
        evidence show that this method is faster than EM
        
        Parameters:
        -----------
        
        max_iter: int
              Maximum number of iterations
              
        
        
        '''
        # number of observations and number of paramters in model
        n,m         = np.shape(self.X)
        
        # initial values of alpha and beta 
        alpha, beta = np.random.random(2)
        
        for i in range(max_iter):
            # find mean for posterior of w and gamma*
            d       =  self.d**2
            gamma   =  np.sum(beta*d/(beta*d + alpha))
            p1_mu   =  np.dot(self.v.T, np.diag(self.d/(d+alpha/beta)))
            p2_mu   =  np.dot(self.u.T,Y)
            mu      =  np.dot(p1_mu,p2_mu)
            
            # store parameters to calculate change later
            alpha_prev = alpha
            beta_prev  = beta
            
            # use updated mu and gamma parameters to update alpha and beta
            alpha      =  gamma/np.dot(mu,mu)
            error      =  self.Y - np.dot(self.X,mu)
            beta       =  (gamma - n)/np.dot(error,error)
            
            # change in parameters
            delta_alpha = abs(alpha_prev - alpha)
            delta_beta  = abs(beta_prev  - beta)
            
            # if change in parameters is less than threshold stop iterations
            if delta_alpha < self.thresh and delta_beta < self.thresh:
                break
                
        return [alpha,beta]
            
            
    def fit(self, evidence_approx_method="fixed_point"):
        '''
        Fits Bayesian linear regression
        '''
        # use type II maximum likelihood to find hyperparameters alpha and beta
        if evidence_approx_method == "fixed_point":
            alpha, beta = self._fixed_point_evidence_approx(100)
        elif evidence_approx_method == "EM":
            alpha, beta = self._em_evidence_approx(100)
            
        # find parameters of posterior distribution
        w_mu, w_beta = self._weights_posterior_params(alpha,beta)
        
            
    def predict_prob(self,Y,X):
        pass
    
    def predict_mean(self):
        pass
    
if __name__=="__main__":
    X      = np.ones([100,1])
    X[:,0] = np.linspace(1,10,100)
    Y      = 4*X+5*np.random.random(100)
    
    
    
    
    
    