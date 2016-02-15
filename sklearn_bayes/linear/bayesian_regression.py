# -*- coding: utf-8 -*-

import numpy as np
from scipy.linalg import pinvh
from scipy.linalg import svd
import warnings

from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.utils import check_X_y



class BayesianRegression(RegressorMixin,LinearModel):
    '''
    Bayesian Regression with type II maximum likelihood for determining point estimates
    for precision variables alpha and beta, where alpha is precision of prior of weights
    and beta is precision of likelihood
    
    Parameters:
    -----------  
    n_iter: int, optional (DEFAULT = 300)
       Maximum number of iterations
         
    tol: float, optional (DEFAULT = 1e-3)
       Threshold for convergence
       
    optimizer: str, optional (DEFAULT = 'fp')
       Method for optimization , either Expectation Maximization or 
       Fixed Point Gull-MacKay {'em','fp'}
       
    fit_intercept: bool, optional (DEFAULT = True)
       If True includes bias term in model
       
    lambda_0: float (DEAFAULT = 1e-10)
       Prevents overflow of precision parameters (this is smallest value RSS can have).
       ( !!! Note if using EM instead of fixed-point, try smaller values
             of lambda_0, for better estimates of variance of predictive distribution )
       
    alpha: float (DEFAULT = 1e-6)
       Initial value of precision paramter for coefficients ( by default we define 
       very broad distribution )
       
       
    Attributes
    ----------
    coef_  : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)
        
    alpha_ : float
        Estimated precision of coefficients
       
    beta_  : float 
        Estimated precision of noise

    sigma_ : array, shape = (n_features, n_features)
        Estimated covariance matrix of the coefficients

    scores_: list
        Values of log-likelihood

    '''
    
    def __init__(self,n_iter = 300, tol = 1e-3, optimizer = 'fp', 
                 fit_intercept = True, lambda_0 = 1e-5, alpha = 1,
                 copy_X = True, verbose = False):
        self.n_iter        =  n_iter
        self.tol           =  tol
        if optimizer not in ['em','fp']:
            raise ValueError('Optimizer can be either "em" of "fp" ')
        self.optimizer     =  optimizer 
        self.fit_intercept =  fit_intercept
        self.alpha         =  alpha 
        #self.beta          =  beta
        self.lambda_0      =  lambda_0        
        self.perfect_fit   =  False
        self.copy_X        =  copy_X
        self.verbose       =  verbose

            
    def fit(self, X, y, evidence_approx_method="fp",max_iter = 100):
        '''
        Fits Bayesian linear regression, returns posterior mean and preision 
        of parameters
        
        Parameters
        ----------
        X: array-like of size [n_samples,n_features]
           Matrix of explanatory variables (should not include bias term)
       
        Y: array-like of size [n_features]
           Vector of dependent variables.
           
        Returns
        -------
        object: self
          self
    
        '''
        # preprocess data
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        n_samples, n_features = X.shape
        X, y, X_mean, y_mean, X_std = self._center_data(X, y, self.fit_intercept,
                                                        self.copy_X)
        self._x_mean_  = X_mean
        self._y_mean_  = y_mean
        self._x_std_   = X_std
        self.scores_   = [np.NINF]
        
        #  precision of noise & and coefficients
        alpha   =  self.alpha
        var_y  = np.var(y)
        # check that variance is non zero !!!
        if var_y == 0 :
            beta = 1e-2
        else:
            beta = 1. / np.var(y)

        # to speed all further computations save svd decomposition and reuse it later
        u,d,v   = svd(X, full_matrices = False)
        Uy      = np.dot(u.T,y)
        dsq     =  d**2

    
        for i in range(self.n_iter):
            
            # find mean for posterior of w ( for EM this is E-step)
            p1_mu   =  v.T * ( d/(dsq+alpha/beta) )
            mu      =  np.dot(p1_mu,Uy)
        
            # precompute errors, since both methods use it in estimation
            error   = y - np.dot(X,mu)
            sqdErr  = np.dot(error,error)
            
            if sqdErr / n_samples < self.lambda_0:
                self.perfect_fit = True
                warnings.warn( ('Almost perfect fit!!! Estimated values of variance '
                                'for predictive distribution are computed using only '
                                'Residual Sum of Squares, terefore they do not increase '
                                'in case of extrapolation')
                             )
                break
            
            if self.optimizer == "fp":           
                gamma      =  np.sum(dsq/(dsq + alpha/beta))
                # use updated mu and gamma parameters to update alpha and beta
                alpha      =   gamma  / np.dot(mu,mu) 
                beta       =  ( n_samples - gamma ) / sqdErr
            else:             
                # M-step, update parameters alpha and beta to maximize ML TYPE II
                alpha      = n_features / (np.dot(mu,mu) + np.sum(1/(beta*dsq+alpha)))
                beta       = n_samples / ( sqdErr + np.sum(dsq/(beta*dsq + alpha))  )
                
            # calculate log likelihood p(Y | X, alpha, beta) (constants are not included)
            normaliser  =  0.5 * ( n_features*np.log(alpha) + n_samples*np.log(beta) )
            normaliser -=  0.5 * np.sum(np.log(beta*dsq+alpha))
            log_like    =  normaliser - 0.5*alpha*np.sum(mu**2) 
            log_like   -=  0.5*beta*sqdErr - 0.5*n_samples*np.log(2*np.pi)         
            self.scores_.append(log_like)
            
            if self.verbose:
                print( ("Iteration {0} completed, value of log " 
                        "likelihood is {1}".format(i,log_like) )
                     ) 

            # if change in log-likelihood is smaller than threshold terminate
            converged = ( self.scores_[-1] - self.scores_[-2] < self.tol)
            if converged or i==self.n_iter -1:
                break

        # pinvh is used for numerical stability (inverse has clased form solution)
        self.sigma_ = pinvh( np.dot(v.T * (beta*dsq + alpha), v) )
        self.coef_  = beta*np.dot(self.sigma_, np.dot(X.T,y))
        self._set_intercept(X_mean,y_mean,X_std)
        self.beta_  = beta
        self.alpha_ = alpha
        return self
            

    def predict_dist(self,X):
        '''
        Calculates  mean and variance of predictive distribution at each data 
        point of test set.
        
        Parameters
        ----------
        x: array-like of size [n_test_samples, n_features]
            Set of features for which corresponding responses should be predicted

        
        Returns
        -------
        :list of two numpy arrays (each has size [n_test_samples])
            Parameters of univariate gaussian distribution [mean and variance] 
        
        '''
        y_hat  = self._decision_function(X)
        x      = (X - self._x_mean_) / self._x_std_
        var_pred  =  1./self.beta_ + np.sum( np.dot( x, self.sigma_ )* x, axis = 1)
        return [y_hat,var_pred]
