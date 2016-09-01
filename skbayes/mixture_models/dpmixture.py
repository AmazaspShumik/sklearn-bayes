from scipy.special import psi
from scipy.misc import logsumexp
from sklearn.base import BaseEstimator
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
import numpy as np
from utils import (BernoulliMixture, GaussianMixture, PoissonMixture, 
                  _e_log_beta)



class DPExponentialMixture(BaseEstimator):
    '''
    Base class for Dirichlet Process Mixture (conjugate exponential family) 
    '''
    
    def __init__(self,n_components,alpha,n_iter,tol, n_init):
        self.n_components = n_components
        self.alpha   = alpha
        self.n_iter  = n_iter
        self.tol     = tol
        self.scores_ = [np.NINF]
        self.n_init  = n_init
    
    
    def _update_sbp(self, resps, Nk):
        '''
        Update parameters of stick breaking represenation of Dirichlet Process
        '''
        a = 1 + Nk
        qz_cum = np.sum(resps,axis = 1, keepdims = True) - np.cumsum(resps,1) 
        b = self.alpha + np.sum(qz_cum,0)
        print b.shape
        return a,b
        
        
    def _update_resps(self,log_pr_x,a,b):
        '''
        Update log of responsibilities
        '''
        psi_ab     = psi(a+b)
        psi_b_ab   = psi(b) - psi_ab  
        pz_cum     = np.cumsum(psi(b) - psi_ab) - psi_b_ab
        log_resps  = log_pr_x + psi(a) - psi_ab + pz_cum
        log_like   = np.copy(log_resps) # = E q_v,q_theta [ logP(X|Z,Theta) + logP(Z|V) ]
        log_resps -= logsumexp(log_resps, axis = 1, keepdims = True)
        resps      = np.exp(log_resps) # = q(Z) - approximating dist of latent var
        # compute part of lower bound that includes mixing latent variable
        # E q_z [ E q_v,q_theta [ logP(X,Z|V,Theta) - log q(Z) ]]
        delta_ll = np.sum(resps*log_like) - np.sum(resps*log_resps)
        return np.exp(log_resps), delta_ll
        
        
    def _fit_single_init(self,X):
        '''
        Fit Dirichlet Process Mixture Model for Exponential Family Distribution
        '''
        # initialise parameters
        # parameters 
        params = self._init_params(X)
        # parameters of beta distribution in stick breaking process
        a = np.ones(self.n_components)
        b = self.alpha * np.ones(self.n_components)
        a0,b0 = np.copy(a), np.copy(b)
        scores = []
        
        for i in xrange(self.n_iter):
            
            log_pr_x = self._log_prob_x(X,params)
            
            # compute q(Z) - approximation of posterior for latent variable
            resps, delta_ll = self._update_resps(log_pr_x,a,b)
            Nk = np.sum(resps,0)
            
            # compute lower bound
            e_logPV = _e_log_beta(a0,b0,a,b)
            e_logQV = _e_log_beta(a,b,a,b)
            # lower bound for difference between prior and approx dist of
            # stick breaking process
            lower_bound_sbp = e_logPV - e_logQV
            scores.append(self._lower_bound(X,delta_ll,params, lower_bound_sbp))
            
            # compute q(V) - approximation of posterior for Stick Breaking Process
            a,b = self._update_sbp(resps,Nk)
            
            # compute q(PARAMS) - approximation of posterior for parameters of 
            # likelihood
            params = self._update_params(X,Nk,resps,params)
            
        return a,b,params,scores
            
            
    def _fit(self,X):
        '''
        Fit parameters
        '''   
        X = self._check_X(X)
        a_,b_,params_ = None,None,None
        scores_ = [np.NINF]
        for i in xrange(self.n_init):
            a,b, params, scores = self._fit_single_init(X)
            if scores_[-1] < scores[-1]:
                a_, b_, params_, scores_ = a,b,params,scores
        return a_, b_, params_, scores_
        
    
        
    def predict_proba(self,X):
        '''
        Predict probability of cluster for test data
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Data Matrix for test data
        
        Returns
        -------
        probs : array, shape = (n_samples,n_components) 
            Probabilities of components membership
        '''
        check_is_fitted(self,'_model_params_')
        X = self._check_X(X)
        log_pr_x = self._log_prob_x(X,self._model_params_)
        a,b = self._sbp_params_
        probs = self._update_resps(log_pr_x,a,b)[0]
        return probs
        
    
    def predict(self,X):
        '''
        Predict cluster for test data
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
           Data Matrix
        
        Returns
        -------
        : array, shape = (n_samples,) component memberships
           Cluster index
        '''
        return np.argmax(self.predict_proba(X),1)
    
    
    def score(self,X):
        '''
        Computes the log probability under the model
        
        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point
            
        Returns
        -------
        logprob: array with shape [n_samples,]
            Log probabilities of each data point in X
        '''
        check_is_fitted(self,'_model_params_')
        pass
        
        

    # abstract methods that need to be implemented in subclass
    def _log_prob_x(self,X,params):
        raise NotImplementedError
        
        
    def _update_params(self, X, Nk, resps, params):
        raise NotImplementedError
    
    
    def _lower_bound(self,X,delta_ll):
        raise NotImplementedError
              
            
    
class DPBMM(DPExponentialMixture, BernoulliMixture):
    '''
    Dirichlet Process Bernoulli Mixture Model
    
    
    Parameters
    ----------
    n_components : int
        Number of mixture components
        
    alpha: float, optional (DEFAULT = 0.1)
        Concentration parameter for Dirichlet Process prior
    '''
    
    def __init__(self, n_components, alpha = 0.1, n_iter = 100, tol = 1e-3, n_init = 3,
                 init_params = None, a = 1, b = 1):
        super(DPBMM,self).__init__(n_components,alpha,n_iter,tol,n_init)
        if init_params is None:
            init_params = {}
        self.init_params = init_params
        self.a   = a
        self.b   = b
        self.delta_ll = []
        self.lower_bound_sbp = []
        self.logPM = []

        
    def _log_prob_x(self,X,params):
        '''
        Expectation of log p(X|Z,Theta) with respect to approximating
        distribution of Theta
        '''
        c = params['c']
        d = params['d']
        psi_cd = psi(c+d)
        x_log = safe_sparse_dot(X,(psi(c)-psi(d)))
        log_probs = x_log + np.sum(psi(d)-psi_cd,axis=0,keepdims = True)
        return log_probs
        
              
    def _update_params(self, X, Nk, resps, params):
        '''
        Update parameters of prior distribution for Bernoulli Succes Probabilities
        '''
        XR = safe_sparse_dot(X.T,resps)
        params['c']  = params['c_init'] + XR
        params['d']  = params['d_init'] + (Nk - XR)
        return params
        
        
    def _lower_bound(self, X, delta_ll, params, lower_bound_sbp):
        ''' 
        Computes lower bound
        '''
        c0,d0,c,d = params['c_init'], params['d_init'], params['c'], params['d']
        e_logPM = _e_log_beta(c0,d0,c,d)
        e_logQM = _e_log_beta(c,d,c,d)
        ll = delta_ll + lower_bound_sbp + e_logPM - e_logQM
        self.delta_ll.append(delta_ll)
        self.lower_bound_sbp.append(lower_bound_sbp)
        self.logPM.append(e_logPM - e_logQM)
        return ll
        
        
    def fit(self,X):
        X = self._check_X(X)
        a_, b_, params_, self.scores_ = self._fit(X)
        # parameters of stick breaking process
        self._sbp_params_ = (a_,b_)
        self._model_params_ = params_
        self.means_ = params_['c'] / ( params_['c'] + params_['d'] )
        return self
        
        
              
class DPPMM(DPExponentialMixture):
    '''
    Dirichlet Process Poisson Mixture Model
    '''
    
    def __init__(self, n_components, alpha = 0.1, n_iter = 100, tol = 1e-3, n_init = 3,
                 init_params = None, a = 1, b = 1):
        super(DPBMM,self).__init__(n_components,alpha,n_iter,tol,n_init)
        if init_params is None:
            init_params = {}
        self.init_params = init_params
        
        
class DPGMM(DPExponentialMixture):
    '''
    Dirichlet Process Gaussian Mixture Model
    '''
    
    def __init__(self, n_components, alpha = 0.1, n_iter = 100, tol = 1e-3, n_init = 3,
                 init_params = None):
        super(DPGMM, self).__init__()
    
        
                
  
      
if __name__ == "__main__":
    dpbmm = DPBMM(n_components = 20, n_iter = 100, alpha = 1)#, init_params = {'a':np.random.random([3,2]),
                                   #                'b':np.random.random([3,2])})
    X = np.zeros([200,3])
    X[0:100,0] = 1
    X[100:200,1] = 1
    #a,b,params, scores = dpbmm._fit(X)
    dpbmm.fit(X)
    y_prob = dpbmm.predict_proba(X)
    y_hat = dpbmm.predict(X)
    #means = params['c'] / ( params['c'] + params['d'] )  
    