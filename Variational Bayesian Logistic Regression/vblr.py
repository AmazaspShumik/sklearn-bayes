# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import expit
from scipy.linalg import pinvh
from sklearn.utils.multiclass import check_classification_targets
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y


#----------------------- Helper functions ----------------------------------


def lam(eps):
    ''' Calculates lambda eps '''
    return 0.5 / eps * ( expit(eps) - 0.5 )
    
#---------------------------------------------------------------------------


class VariationalLogisticRegression(LinearClassifierMixin, BaseEstimator):
    '''
    Variational Bayesian Logistic Regression 
    
    Parameters:
    -----------
    n_iter: int, optional (DEFAULT = 300 )
       Maximum number of iterations
       
    tol: float, optional (DEFAULT = 1e-3)
       Convergence threshold, if cange in coefficients is less than threshold
       algorithm is terminated
    
    fit_intercept: bool, optinal ( DEFAULT = True )
       If True uses bias term in model fitting
       
    a: float, optional (DEFAULT = 1e-6)
       Rate parameter for Gamma prior on precision parameter of coefficients
       
    b: float, optional (DEFAULT = 1e-6)
       Shape parameter for Gamma prior on precision parameter of coefficients
    
    verbose: bool, optional (DEFAULT = False)
       Verbose mode
       
       
    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)

    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients
    
    intercept_: array, shape = (n_features)
        intercepts
        

    References:
    -----------
    Bishop 2006, Pattern Recognition and Machine Learning ( Chapter 10 )
    Murphy 2012, Machine Learning A Probabilistic Perspective ( Chapter 21 )
    '''
    def __init__(self,  n_iter = 300, tol = 1e-3, fit_intercept = True,
                 a = 1e-6, b = 1e-6, verbose = True):
        self.n_iter            = n_iter
        self.tol               = tol
        self.verbose           = verbose
        self.fit_intercept     = fit_intercept
        self.a                 =  a
        self.b                 =  b
        
        
    def fit(self,X,y):
        '''
        Fits variational Bayesian Logistic Regression
        
        Parameters
        ----------
        X: array-like of size [n_samples, n_features]
           Matrix of explanatory variables
           
        y: array-like of size [n_samples]
           Vector of dependent variables

        Returns
        -------
        self: object
           self
        '''
        # preprocess data
        X,y = check_X_y( X, y , dtype = np.float64)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # take into account bias term if required 
        n_samples, n_features = X.shape
        n_features = n_features + int(self.fit_intercept)
        if self.fit_intercept:
            X = np.hstack( (np.ones([n_samples,1]),X))
        
        # handle multiclass problems using One-vs-Rest 
        if n_classes < 2:
            raise ValueError("Need samples of at least 2 classes")
        if n_classes > 2:
            self.coef_, self.sigma_ = [0]*n_classes,[0]*n_classes
            self.intercept_         = [0]*n_classes
        else:
            self.coef_, self.sigma_, self.intercept_ = [0],[0],[0]
        
        # huperparameters of 
        a  = self.a + 0.5 * n_features
        b  = self.b
        
        for i in range(len(self.coef_)):
            if n_classes == 2:
                pos_class = self.classes_[1]
            else:
                pos_class   = self.classes_[i]
            mask            = (y == pos_class)
            y_bin           = np.ones(y.shape, dtype=np.float64)
            y_bin[~mask]    = 0
            coef_, sigma_  = self._fit(X,y_bin,a,b)
            intercept_ = 0
            if self.fit_intercept:
                intercept_  = coef_[0]
                coef_       = coef_[1:]
            self.coef_[i]   = coef_
            self.intercept_[i] = intercept_
            self.sigma_[i]  = sigma_
        self.coef_  = np.asarray(self.coef_)
        return self
        

    def predict_proba(self,x):
        '''
        Predicts probabilities of targets for test set
        
        Parameters
        ----------
        X: array-like of size [n_samples_test,n_features]
           Matrix of explanatory variables (test set)
           
        Returns
        -------
        probs: numpy array of size [n_samples_test]
           Estimated probabilities of target classes
        '''
        scores = self.decision_function(x)
        if self.fit_intercept:
            x = np.hstack( (np.ones([x.shape[0],1]),x))
        sigma  = np.asarray([np.sum(np.dot(x,s)*x,axis = 1) for s in self.sigma_])
        ks = 1. / ( 1. + np.pi*sigma / 8)**0.5
        probs = expit(scores.T*ks).T
        if probs.shape[1] == 1:
            probs =  np.hstack([1 - probs, probs])
        else:
            probs /= np.reshape(np.sum(probs, axis = 1), (probs.shape[0],1))
        return probs

            
    def _fit(self,X,y,a,b):
        '''
        Fits single classifier for each class (for OVR framework)
        '''
        eps = 1
        XY  = np.dot( X.T , (y-0.5))
        w0  = np.zeros(X.shape[1])
  
        for i in range(self.n_iter):
            # In the E-stpe we update approximation of 
            # posterior distribution q(w,alpha) = q(w)*q(alpha)
            
            # --------- update q(w) ------------------
            l  = lam(eps)
            w,sigma = self._posterior_dist(X,l,a,b,XY)
            
            
            # -------- update q(alpha) ---------------
            
            E_w_sq = np.outer(w,w) + sigma
            b = self.b + np.sum(w**2) + np.trace(sigma)#0.5*np.trace(E_w_sq)
            
            # In the M-step we update parameter eps which controls 
            # accuracy of local variational approximation
            eps = np.sqrt( np.sum( np.dot(X,E_w_sq)*X, axis = 1))
            
            # convergence
            if np.sum(abs(w-w0) > self.tol) == 0 or i==self.n_iter-1:
                break
            w0 = w
            
        l  = lam(eps)
        coef_, sigma_  = self._posterior_dist(X,l,a,b,XY)

        return coef_, sigma_


    def _posterior_dist(self,X,l,a,b,XY):
        '''
        Finds gaussian approximation to posterior of coefficients
        '''
        sigma_inv  = 2*np.dot(X.T*l,X)
        alpha_vec  = np.ones(X.shape[1])*float(a) / b
        if self.fit_intercept:
            alpha_vec[0] = 0
        np.fill_diagonal(sigma_inv, np.diag(sigma_inv) + alpha_vec)
        sigma_   = pinvh(sigma_inv)
        mean_    = np.dot(sigma_,XY)     
        return [mean_, sigma_]

