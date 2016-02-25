# -*- coding: utf-8 -*-
from sklearn.base import BaseEstimator
from scipy.special import psi,gammaln
from scipy.misc import logsumexp
from scipy.linalg import pinvh
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import csr_matrix,isspmatrix
from sklearn.cluster import KMeans
import numpy as np


#============================= Helpers =============================================#


class StudentMultivariate(object):
    '''
    Multivariate Student Distribution
    '''
    def __init__(self,mean,precision,df,d):
        self.mu   = mean
        self.L    = precision
        self.df   = df
        self.d    = d
                
                
    def logpdf(self,x):
        '''
        Calculates value of logpdf at point x
        '''
        xdiff     = x - self.mu
        quad_form = np.sum( np.dot(xdiff,self.L)*xdiff, axis = 1)
        
        
        return ( gammaln( 0.5 * (self.df + self.d)) - gammaln( 0.5 * self.df ) +
                 0.5 * np.linalg.slogdet(self.L)[1] - 0.5*self.d*np.log( self.df*np.pi) -
                 0.5 * (self.df + self.d) * np.log( 1 + quad_form / self.df )
               )
        
    def pdf(self,x):
        '''
        Calculates value of pdf at point x
        '''
        return np.exp(self.logpdf(x))
        
        

def _e_log_dirichlet(alpha0,alphaK):
    ''' Calculates expectation of log pdf of dirichlet distributed parameter '''
    log_C   = gammaln(np.sum(alpha0)) - np.sum(gammaln(alpha0))
    e_log_x = np.dot(alpha0-1,psi(alphaK) - psi(np.sum(alphaK)))
    return np.sum(log_C + e_log_x)


def _e_log_beta(c0,d0,c,d):
    ''' Calculates expectation of log pdf of beta distributed parameter'''
    log_C    = gammaln(c0 + d0) - gammaln(c0) - gammaln(d0)
    psi_cd   = psi(c+d)
    log_mu   = (c0 - 1) * ( psi(c) - psi_cd )
    log_i_mu = (d0 - 1) * ( psi(d) - psi_cd )
    return np.sum(log_C + log_mu + log_i_mu)
    

def _get_classes(X):
    '''Finds number of unique elements in matrix'''
    if isspmatrix(X):
        v = X.data
        if len(v) < X.shape[0]*X.shape[1]:
            v = np.hstack((v,np.zeros(1)))
        V     = np.unique(v)
    else:
        V     = np.unique(X)
    return V
    
   
#==================================================================================#
   

class GeneralMixtureModelExponential(BaseEstimator):
    '''
    Superclass for Mixture Models 
    '''
    def __init__(self, n_components = 2, n_iter = 100, tol = 1e-3, 
                 alpha0 = 10, n_init = 3, init_params = None,
                 compute_score = False, verbose = False):
        self.n_iter              = n_iter
        self.n_init              = n_init
        self.n_components        = n_components
        self.tol                 = tol
        self.alpha0              = alpha0
        self.compute_score       = compute_score
        self.init_params         = init_params
        self.verbose             = verbose
        
    
    def _update_resps(self, X, alphaK, *args):
        '''
        Updates distribution of latent variable with Dirichlet prior
        '''
        e_log_weights = psi(alphaK) - psi(np.sum(alphaK))
        return self._update_resps_parametric(X,e_log_weights,self.n_components,
                                             *args)

        
    def _update_resps_parametric(self, X, log_weights, clusters, *args):
        ''' Updates distribution of latent variable with parametric weights'''
        log_resps  = np.asarray([self._update_logresp_cluster(X,k,log_weights,*args)
                                for k in range(clusters)]).T
        log_like       = np.copy(log_resps)
        log_resps     -= logsumexp(log_resps, axis = 1, keepdims = True)
        resps          = np.exp(log_resps)
        delta_log_like = np.sum(resps*log_like) - np.sum(resps*log_resps)
        return resps, delta_log_like
        
        
    def _update_dirichlet_prior(self,alpha_init,Nk):
        '''
        For all models defined in this module prior for cluster distribution 
        is Dirichlet, so all models will need to update parameters
        '''
        return alpha_init + Nk
        
        
    def _check_X(self,X):
        '''
        Checks validity of input for all mixture models
        '''
        X  = check_array(X, accept_sparse = ['csr'])
        # check that number of components is smaller or equal to number of samples
        if X.shape[0] < self.n_components:
            raise ValueError(('Number of components should not be larger than '
                              'number of samples'))
        return X
        
    
    def _check_convergence(self,metric_diff,n_params):
        ''' Checks convergence of mixture model'''
        convergence = metric_diff / n_params < self.tol
        if self.verbose and convergence:
            print("Algorithm converged")
        return convergence
    
    
    def predict(self,X):
        '''
        Predict cluster for test data
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
        
        Returns
        -------
        C : array, shape = (n_samples,) component memberships
        '''
        return np.argmax(self.predict_proba(X),1)
  

#==================================================================================#


class VBBMM(GeneralMixtureModelExponential):
    ''' 
    Variational Bayesian Bernoulli Mixture Model
    
    Parameters
    ----------
    n_components : int, optional (DEFAULT = 2)
        Number of mixture components
        
    n_init :  int, optional (DEFAULT = 5)
        Number of restarts of algorithm
        
    n_iter : int, optional (DEFAULT = 100)
        Number of iterations of Mean Field Approximation Algorithm

    tol : float, optional (DEFAULT = 1e-3)
        Convergence threshold
        
    alpha0 :float, optional (DEFAULT = 1)
        Concentration parameter for Dirichlet prior on weights
        
    c : float , optional (DEFAULT = 1)
        Shape parameter for beta distribution
        
    d: float , optional (DEFAULT = 1)
        Shape parameter for beta distribution
        
    compute_score: bool, optional (DEFAULT = True)
        If True computes logarithm of lower bound at each iteration

    verbose : bool, optional (DEFAULT = False)
        Enable verbose output
        
        
    Attributes
    ----------
    weights_ : numpy array of size (n_components,)
        Mixing probabilities for each cluster
        
    means_ : numpy array of size (n_features, n_components)
        Mean success probabilities for each cluster
        
    scores_: list of unknown size (depends on number of iterations)
        Log of lower bound

    '''
    def __init__(self, n_components = 2, n_init = 3, n_iter = 100, tol = 1e-3, 
                 alpha0 = 1, c = 1e-2, d = 1e-2, init_params = None,
                 compute_score = False, verbose = False):
        super(VBBMM,self).__init__(n_components,n_iter,tol,alpha0, n_init,
                                   init_params, compute_score, verbose)
        self.c = c
        self.d = d
        
    
    def _check_X_train(self,X):
        ''' Preprocesses & check validity of training data'''
        X                 = super(VBBMM,self)._check_X(X)
        self.classes_     = _get_classes(X)
        n                 = len(self.classes_)
        # check that there are only two categories in data
        if n != 2:
            raise ValueError(('There are {0} categorical values in data, '
                               'model accepts data with only 2'.format(n)))
        return 1*(X==self.classes_[1])
        
        
    def _check_X_test(self,X):
        ''' Preprocesses & check validity of test data'''
        X = check_array(X, accept_sparse = ['csr'])
        classes_   = _get_classes(X)
        n          = len(classes_)
        # check number of classes 
        if n != 2:
            raise ValueError(('There are {0} categorical values in data, '
                               'model accepts data with only 2'.format(n))) 
        # check whether these are the same classes as in training
        if classes_[0]==self.classes_[0] and classes_[1] == self.classes_[1]:
            return 1*(X==self.classes_[1])
        else:
            raise ValueError(('Classes in training and test set are different, '
                              '{0} in training, {1} in test'.format(self.classes_,
                              classes_)))
        

    def _fit(self,X):
        '''
        Performs single run of VBBMM
        '''
        n_samples, n_features = X.shape
        n_params              = n_features*self.n_components + self.n_components
        scores                = []    

        # use initial values of hyperparameter as starting point
        c = self.c * np.random.random([n_features,self.n_components])
        d = self.d * np.random.random([n_features,self.n_components])
        c_old, d_old = c,d
        c_prev,d_prev = c,d
        
        # we need to break symmetry for mixture weights
        alphaK      = self.alpha0*np.random.random(self.n_components)
        alphaK_old  = alphaK
        alphaK_prev = alphaK
        
        for i in range(self.n_iter):
            
            # ---- update approximating distribution of latent variable ----- #
            
            resps, delta_log_like = self._update_resps(X,alphaK,c,d)
            
            # reuse responsibilities in computing lower bound
            if self.compute_score:
                scores.append(self._compute_score(delta_log_like, alphaK_old, 
                                                  alphaK, c_old, d_old, c, d))
            
            # ---- update approximating distribution of parameters ---------- #
            
            Nk     = sum(resps,0)
            
            # update parameters of Dirichlet Prior
            alphaK = self._update_dirichlet_prior(alphaK_old,Nk)
    
            # update parameters of Beta distributed success probabilities
            c,d    = self._update_params( X, Nk, resps)
            diff   = np.sum(abs(c-c_prev) + abs(d-d_prev) + abs(alphaK-alphaK_prev))
            
            if self.verbose:
                if self.compute_score:
                    print('Iteration {0}, value of lower bound is {1}'.format(i,scores[-1]))
                else:
                    print(('Iteration {0}, normalised delta of parameters ' 
                          'is {1}').format(i,diff))

            if self._check_convergence(diff,n_params):
                break
            c_prev,d_prev = c,d
            alphaK_prev   = alphaK
            
        # compute log of lower bound to compare best model
        resps, delta_log_like = self._update_resps(X,alphaK,c,d)
        scores.append(self._compute_score(delta_log_like, alphaK_old,  
                                          alphaK, c_old, d_old, c, d))     
        return alphaK, c, d, scores


    def _update_logresp_cluster(self,X,k,e_log_weights,*args):
        '''
        Unnormalised responsibilities for single cluster
        '''
        c,d   = args
        ck,dk = c[:,k], d[:,k]
        xcd   = safe_sparse_dot(X , (psi(ck) - psi(dk)))
        log_resp = xcd + np.sum(psi(dk) - psi(ck + dk)) + e_log_weights[k]
        return log_resp
        
        
    def _update_params(self,X,Nk,resps):
        '''
        Update parameters of prior distribution for Bernoulli Succes Probabilities
        '''
        XR = safe_sparse_dot(X.T,resps)
        c  = self.c + XR
        d  = self.d + (Nk - XR)
        return c,d
       
       
    def _compute_score(self, delta_log_like, alpha_init, alphaK, c_old, d_old, c, d):
        '''
        Computes lower bound
        '''
        log_weights_prior   =  _e_log_dirichlet(alpha_init, alphaK)
        log_success_prior   =  _e_log_beta(c_old,d_old,c,d)
        log_weights_approx  = -_e_log_dirichlet(alphaK,alphaK)
        log_success_approx  = -_e_log_beta(c,d,c,d)
        lower_bound         =  log_weights_prior
        lower_bound        +=  log_success_prior   + log_weights_approx
        lower_bound        +=  log_success_approx  + delta_log_like
        return lower_bound
        

    def fit(self,X):
        '''
        Fits Variational Bayesian Bernoulli Mixture Model
        
        Parameters
        ----------
        X: array-like or sparse csr_matrix of size [n_samples, n_features]
           Data Matrix
           
        Returns
        -------
        self: object
           self
         
        Practical Advice
        ----------------
        Significant speedup can be achieved by using sparse matrices
        (see scipy.sparse.csr_matrix)
        
        '''
        # preprocess data
        X = self._check_X_train(X)
        
        # refit & choose best model (log of lower bound is used)
        score_old        = [np.NINF]
        alpha_, c_ , d_  = 0,0,0
        for j in range(self.n_init):
            if self.verbose:
                print("New Initialisation, restart number {0} \n".format(j))
            alphaK, c, d, score = self._fit(X)
            if score[-1] > score_old[-1]:
                alpha_ , c_ , d_ = alphaK, c, d
                score_old        = score
        
        # save parameters corresponding to best model
        self.alpha_      = alpha_
        self.means_      = c_ / (c_ + d_)
        self.c_, self.d_ = c_,d_
        self.weights_    = alpha_ / np.sum(alpha_)
        self.scores_     = score_old
        return self
        

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
        check_is_fitted(self,'scores_')
        X = self._check_X_test(X)
        probs = self._update_resps(X,self.alpha_,self.c_, self.d_)[0]
        return probs
    
    
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
        probs = self.predict_proba(X)
        return np.log(np.dot(probs,self.weights_))
        
        
    def cluster_prototype(self):
        pass


#==================================================================================#


class VBMMM(GeneralMixtureModelExponential):
    '''
    Variational Bayesian Multinomial Mixture Model

    
    Parameters
    ----------
    n_components : int, optional (DEFAULT = 2)
        Number of mixture components
        
    n_init :  int, optional (DEFAULT = 5)
        Number of restarts of algorithm
        
    n_iter : int, optional (DEFAULT = 100)
        Number of iterations of Mean Field Approximation Algorithm

    tol : float, optional (DEFAULT = 1e-3)
        Convergence threshold
        
    alpha0 :float, optional (DEFAULT = 1)
        Concentration parameter for Dirichlet prior on weights
        
    beta0 : float , optional (DEFAULT = 1)
        Concentration parameter for Dirichlet prior on Multionomial probabilities
        
    precompute_X: bool, optional (DEFAULT = True)
        If True creates list of binary sparse matrices corresponding to each
        unique element in training matrix
        
    compute_score: bool, optional (DEFAULT = True)
        If True computes logarithm of lower bound at each iteration

    verbose : bool, optional (DEFAULT = False)
        Enable verbose output
        
        
    Attributes
    ----------
    weights_ : numpy array of size (n_components,)
        Mixing probabilities for each cluster
        
    means_ : numpy array of size (n_features, n_components)
        Mean Multinomial Probabilities for each cluster
        
    scores_: list of unknown size (depends on number of iterations)
        Log of lower bound
    
    '''
    def __init__(self, n_components = 2, n_init = 5, n_iter = 100, tol = 1e-3, 
                 alpha0 = 10, beta0 = 10 ,init_params = None, precompute_X = True,
                 compute_score = False, verbose = False):
        super(VBMMM,self).__init__(n_components,n_iter,tol,alpha0,n_init,
                                   init_params,compute_score, verbose)
        self.beta0        = beta0
        self.precompute_X = precompute_X

   
    def _check_X_test(self,X):
        ''' Preprocesses & check validity of test data'''
        X = check_array(X, accept_sparse = ['csr'])
        classes_   = _get_classes(X)
        n          = len(classes_)
        # check number of unique elements in training and test is the same
        if n != len(self.classes_):
            raise ValueError(('Number of unique elements in training  '
                               'data is {0}, number unique elements in test '
                               'set is {1}'.format(len(self.classes_),n))) 
        # check whether these are the same unique elements as in test data
        if np.prod(self.classes_==classes_)==1:
            return self._precompute_X(X)
        else:
            raise ValueError(('Classes in training and test set are different, '
                              '{0} in training, {1} in test'.format(self.classes_,
                              classes_)))
        
    
    def _precompute_X(self,X):
        '''Precomputes binary matrices '''
        zero_class = csr_matrix(np.ones(X.shape))
        precomputed_X = [0]*len(self.classes_) 
        for i,class_ in enumerate(self.classes_[1:]):
            if isspmatrix(X):
                precomputed_X[i+1] = 1*(X==class_)
            else:
                precomputed_X[i+1] = csr_matrix(1*(X==class_))
            zero_class -= precomputed_X[i+1]
        precomputed_X[0] = zero_class
        return precomputed_X


    def _get_class(self,X):
        '''Generator for binary matrix [True,False] for each class'''
        # TODO: handle zero elements of sparse matrix more efficiently
        for i,class_ in enumerate(self.classes_):
            if self.precompute_X:
                yield X[i]
            else:
                if isspmatrix(X):
                   yield 1*(X==class_)
                else:
                   yield csr_matrix(1*(X==class_))
                

    def fit(self,X):
        '''
        Fits Variational Bayesian Multinomial Mixture Model
        
        Parameters
        ----------
        X: array-like or sparse csr_matrix of size [n_samples, n_features]
           Data Matrix
           
        Returns
        -------
        self: object
           self
        '''
        # preprocess data
        X = self._check_X(X)
        n_samples,n_features = X.shape
        self.classes_  = _get_classes(X)
        if self.precompute_X:
            X = self._precompute_X(X)

        # refit & choose best model (log of lower bound is used)
        score_old        = [np.NINF]
        alpha_, beta_    = 0,0
        for j in range(self.n_init):
            if self.verbose:
                print("New Initialisation, restart number {0}".format(j))
            alphaK, betaK, score = self._fit(X,n_samples,n_features)
            if score[-1] > score_old[-1]:
                alpha_, beta_ = alphaK, betaK
                score_old     = score
        
        # save parameters corresponding to best model
        self.alpha_      = alpha_
        self.beta_       = beta_
        self.means_      = beta_ / np.sum(beta_, axis = 1, keepdims = 1)
        self.weights_    = alpha_ / np.sum(alpha_)
        self.scores_     = score_old
        return self
        
        
    def _fit(self,X,n_samples,n_features):
        '''
        Fits Variational Multinomial Mixture Model
        '''
        n_classes = len(self.classes_)
        n_params  = self.n_components #+ n_features*n_classes*self.n_components
        alphaK  = self.alpha0*np.random.random(self.n_components)
        betaK   = np.asarray([self.beta0*np.random.random([n_features,n_classes])
                   for k in range(self.n_components)])
        betaK_old  = np.copy(betaK)
        alphaK_old, alphaK_prev  = np.copy(alphaK), alphaK
        scores  = []
        
        for i in range(self.n_iter):
            
            # ---- update approximating distribution of latent variable ----- #
            
            resps, delta_log_like = self._update_resps(X,alphaK,betaK, n_samples)
            
            # compute value of lower bound
            if self.compute_score:
                scores.append(self._compute_score(delta_log_like, alphaK_old, alphaK,
                                                  betaK_old, betaK))
            
            # ---- update approximating distribution of parameters ---------- #
            
            Nk     = sum(resps,0)
            alphaK = self._update_dirichlet_prior(alphaK_old,Nk)
            betaK  = self._update_params(X,Nk, resps, betaK, betaK_old)
            diff   = np.sum(abs(alphaK-alphaK_prev))
            if self.verbose:
                if self.compute_score:
                    print('Iteration {0}, value of lower bound is {1}'.format(i,scores[-1]))
                else:
                    print(('Iteration {0}, normalised delta of parameters ' 
                          'is {1}').format(i,diff))
            if self._check_convergence(diff,n_params):
                break
            alphaK_prev = alphaK
            
        # compute score to find best model
        resps, delta_log_like = self._update_resps(X,alphaK,betaK,n_samples)
        scores.append(self._compute_score(delta_log_like, alphaK_old, alphaK,
                                                  betaK_old, betaK))
        return alphaK, betaK, scores


        
    def _update_logresp_cluster(self,X,k,e_log_weights,*args):
        '''
        Calculates log of unnormalised responsibilities for single cluster
        '''
        betak,n_samples = args
        betak           = betak[k]
        log_resp = np.zeros(n_samples)
        for i,x in enumerate(self._get_class(X)):
            log_resp += safe_sparse_dot(x, psi(betak[:,i]) - psi(np.sum(betak[:,i])))
        log_resp += e_log_weights[k]
        return log_resp
        
        
    def _update_params(self,X,Nk,resps,betaK,betaK_old):
        '''
        Update parameters of distr
        '''
        for i,x in enumerate(self._get_class(X)):
            XR = safe_sparse_dot(x.T,resps)            
            for k in range(self.n_components):
                betaK[k][:,i] = XR[:,k] + betaK_old[k][:,i]
        return betaK
        
       
    def _compute_score(self, delta_log_like, alphaK_old, alphaK, betaK_old, betaK):
        '''
        Computes lower bound
        '''
        log_weights_prior   =  _e_log_dirichlet(alphaK_old, alphaK)
        log_weights_approx  = -_e_log_dirichlet(alphaK,alphaK)
        delta_log_succes    =  0
        for beta_, beta_init in zip(betaK, betaK_old):
            for i in range(beta_.shape[0]):
                log_approx  = -_e_log_dirichlet(beta_[i,:], beta_[i,:])
                log_prior   =  _e_log_dirichlet(beta_init[i,:], beta_[i,:])
                delta_log_succes +=  log_prior + log_approx

        lower_bound         =  log_weights_prior + log_weights_approx
        lower_bound        +=  delta_log_succes  + delta_log_like
        return lower_bound
        
        
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
        check_is_fitted(self,'scores_')
        n_samples = X.shape[0]
        X = self._check_X_test(X)
        probs = self._update_resps(X,self.alpha_,self.beta_,n_samples)[0]
        return probs
    
    
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
        probs = self.predict_proba(X)
        return np.log(np.dot(probs,self.weights_))
        
        
    def cluster_prototype(self):
        '''
        Computes most likely 
        
        Returns
        -------
        
        
        '''
        prototypes = [0]*self.n_components
        for k in range(self.n_components):
            prototypes[k] = self.classes_[np.argmax(self.means_[k],1)]
        return prototypes
        
        
        

                
        
class VBGMMARD(GeneralMixtureModelExponential):
    '''
    Variational Bayeisian Gaussian Mixture Model with Automatic Relevance 
    Determination. Implemented model automatically selects number of relevant 
    components through mixture of Type II Maximum Likelihood and Mean Field
    Approximation.
    This is not fully Bayesian Model, it does not place any prior on weights
    (weighta are assumed to be paramters that needs to be optimized to maximize
    value of lower bound)
    
    Parameters:
    -----------       
    n_components : int, optional (DEFAULT = 10)
       Maximum number of mixture components
       
    tol : float, optional (DEFAULT = 1e-3)
       Convergence threshold
       
    n_iter : int, optional (DEFAULT = 100)
       Maximum number of iterations
       
    n_mfa_iter: int, optional (DEFAULT = 1)
       Maximum number of iterations for Mean Field Approximation of lower bound
       
    n_init: int , optional (DEFAULT = 5)
       Number of restarts in initialization
       
    prune_thresh: float, optional (DEFAULT = 1e-3)
       Threshold for cluster removal. If weight corresponding to cluster becomes
       smaller than threshold it is removed.
     
    init_params: dict, optional (DEFAULT = {})
       Initial parameters for model (keys = ['dof','covar','weights'])
           'dof'    : int  
                 Degrees of freedom for prior distribution
                 
           'covar'  : array of size (n_features, n_features)
                  Inverse of scaling matrix for prior wishart distribution
                  
           'weights': array of size (n_components,) 
                  Latent variable distribution parameter (cluster weights)
       
           'beta'   : float
                  Scaling constant for precision of mean's prior 
                  
           'means'  : array of size (n_components, n_features) 
                  Mean

    verbose: bool, optional (DEFAULT = False)
       Enables verbose output
        
        
    References:
    -----------
    1) Adrian Corduneanu and Chris Bishop, Variational Bayesian Model Selection for 
       Mixture Distributions (2001)
    '''
    
    def __init__(self, n_components = 10, tol = 1e-3, n_iter = 100, n_mfa_iter = 1,
                 n_init = 5, prune_thresh = 1e-3, compute_score = False, 
                 init_params = dict(), verbose = False ):
        super(VBGMMARD,self).__init__(n_components, n_iter, tol,1, n_init,
                                      init_params,compute_score, verbose)
        self.n_mfa_iter   = n_mfa_iter
        self.prune_thresh = prune_thresh
        

    def _init_params(self,X):
        '''
        Initialise parameters
        '''
        d = X.shape[1]

        # initialise prior on means & precision matrices
        if 'means' in self.init_params:
            means0   = self.init_params['means']
        else:
            kms = KMeans(n_init = self.n_init, n_clusters = self.n_components)
            means0 = kms.fit(X).cluster_centers_
            
        if 'covar' in self.init_params:
            scale_inv0 = self.init_params['covar']
            scale0     = pinvh
        else:
            # heuristics to define broad prior over precision matrix
            diag_els   = np.abs(np.max(X,0) - np.min(X,0))
            scale_inv0 = np.diag( diag_els/2  )
            scale0     = np.diag( 1./ diag_els )
            
        if 'weights' in self.init_params:
            weights0  = np.ones(self.n_components) / self.n_components
        else:
            weights0  = np.ones(self.n_components) / self.n_components
          
        if 'dof' in self.init_params:
            dof0 = self.init_params['dof']
        else:
            dof0 = d
            
        if 'beta' in self.init_params:
            beta0 = self.init_params['beta']
        else:
            beta0 = 1e-3
            
        # clusters that are not pruned 
        self.active  = np.ones(self.n_components, dtype = np.bool)
        
        # checks initialisation errors in case parameters are user defined
        assert dof0 >= d,( 'Degrees of freedom should be larger than '
                                'dimensionality of data')
        assert means0.shape[0] == self.n_components,('Number of centrods defined should '
                                                     'be equal to number of components')
        assert means0.shape[1] == d,('Dimensioanlity of means and data '
                                          'should be the same')
        assert weights0.shape[0] == self.n_components,('Number of weights should be '
                                                           'to number of components')
        
        # At first iteration these parameters are equal to priors, but they change 
        # at each iteration of mean field approximation
        scale   = np.array([np.copy(scale0) for _ in range(self.n_components)])
        means   = np.copy(means0)
        weights = np.copy(weights0)
        dof     = dof0*np.ones(self.n_components)
        beta    = beta0*np.ones(self.n_components)
        init_   = [means0, scale0, scale_inv0, beta0, dof0, weights0]
        iter_   = [means, scale, scale_inv0, beta, dof, weights]
        return init_, iter_
        
        
    def fit(self, X):
        '''
        Fits Variational Bayesian GMM with ARD, automatically determines number 
        of mixtures component.
        
        Parameters
        -----------
        X: numpy array [n_samples,n_features]
           Data matrix
           
        Returns
        -------
        self: object
           self
        '''
        X                     =  self._check_X(X)
        n_samples, n_features =  X.shape
        init_, iter_          =  self._init_params(X)
        if self.verbose:
            print('Parameters are initialise ...')
        means0, scale0, scale_inv0, beta0, dof0, weights0  =  init_
        means, scale, scale_inv, beta, dof, weights        =  iter_
        # all clusters are active initially
        active = np.ones(self.n_components, dtype = np.bool)
        self.n_active = np.sum(active)
        
        for j in range(self.n_iter):
            
            means_before = np.copy(means)
            
            # Approximate lower bound with Mean Field Approximation
            for i in range(self.n_mfa_iter):

                # Update approx. posterior of latent distribution 
                resps, delta_ll = self._update_resps_parametric(X, weights, self.n_active,
                                                                dof, means, scale, beta)
                
                # Update approx. posterior of means & pecision matrices
                Nk     = np.sum(resps,axis = 0)
                Xk     = [np.sum(resps[:,k:k+1]*X,0) for k in range(self.n_active)]
                Sk     = [np.dot(resps[:,k]*X.T,X) for k in range(self.n_active)]
                beta, means, dof, scale = self._update_params(Nk, Xk, Sk, beta0, 
                                                              means0, dof0, scale_inv0,
                                                              beta,  means,  dof, scale)
            
            
            # Maximize lower bound with respect to weights 
        
            # update weights to maximize lower bound  
            weights      = Nk / n_samples
            
            # prune all irelevant weights
            active            = weights > self.prune_thresh
            means0            = means0[active,:]
            scale             = scale[active,:,:]
            weights           = weights[active]
            weights          /= np.sum(weights)
            dof               = dof[active]
            beta              = beta[active]
            n_comps_before    = self.n_active
            means             = means[active,:]
            self.n_active     = np.sum(active)
            if self.verbose:
                print(('Iteration {0} completed, number of active clusters '
                       ' is {1}'.format(j,self.n_active)))
                       
            # check convergence
            if n_comps_before == self.n_active:
                if self._check_convergence(n_comps_before,means_before,means):
                    if self.verbose:
                        print("Algorithm converged")
                    break
                        
        self.means_   = means
        self.weights_ = weights
        self.covars_  = np.asarray([1./df * pinvh(sc) for sc,df in zip(scale,dof)])
        # calculate parameters for predictive distribution
        self.predictors_ = self._predict_dist_params(dof,beta,means,scale)
        return self

        
        
    def _update_logresp_cluster(self, X, k, weights, dof, means, scale, beta):
        '''
        Updates responsibilities for single cluster, calculates expectation
        of logdet of precision matrix.
        '''
        d = X.shape[1] 
        # calculate expectation of logdet of precision matrix
        scale_logdet   = np.linalg.slogdet(scale[k])[1]
        e_logdet_prec  = sum([psi(0.5*(dof[k]+1-i)) for i in range(1,d+1)])
        e_logdet_prec += scale_logdet + d*np.log(2)
           
        # calculate expectation of quadratic form (x-mean_k)'*precision_k*(x - mean_k)
        x_diff         = X - means[k,:]
        e_quad_form    = np.sum( np.dot(x_diff,scale[k,:,:])*x_diff, axis = 1 )
        e_quad_form   *= dof[k]
        e_quad_form   += d / beta[k] 
        
        # responsibilities without normalisation
        log_pnk        = np.log(weights[k]) + 0.5*e_logdet_prec - 0.5*e_quad_form
        log_pnk       -= d * np.log( 2 * np.pi)
        return log_pnk

    
    def _update_params(self, Nk, Xk, Sk, beta0, means0, dof0, scale_inv0,
                                 beta, means, dof, scale):
        ''' Updates distribution of means and precisions '''
        for k in range(self.n_active):
            # update mean and precision for each cluster
            beta[k]   = beta0 + Nk[k]
            means[k]  = (beta0*means0[k,:] + Xk[k]) / beta[k]
            dof[k]    = dof0 + Nk[k] + 1
            # precision calculation is ugly but prevent overflow & underflow
            scale[k,:,:]  = pinvh( scale_inv0 + (beta0*Sk[k] + Nk[k]*Sk[k] - 
                                 np.outer(Xk[k],Xk[k]) - 
                                 beta0*np.outer(means0[k,:] - Xk[k],means0[k,:])) /
                                 (beta0 + Nk[k]) )
        return beta,means,dof,scale
                                 
                             
    def _check_convergence(self,n_components_before,means_before, means):
        ''' Checks convergence '''
        conv = True
        for mean_before,mean_after in zip(means_before,means):
            mean_diff = mean_before - mean_after
            conv  = conv and np.sum(np.abs(mean_diff)) / means.shape[1] < self.tol
        return conv
        
        
    def _predict_dist_params(self, dof, beta, means, scale):
        ''' Computes parameters for predictive distribution '''
        d = means.shape[1]
        predictors = []
        for k in range(self.n_active):
            df    = dof[k] + 1 - d
            prec  = scale[k,:,:] * beta[k] * df / (1 + beta[k])
            predictors.append(StudentMultivariate(means[k,:],prec,dof[k],d))
        return predictors
        
        
    def predict_proba(self,X):
        '''
        Calculates 
        
        Parameters
        ----------
        
        '''
        X = check_array(X)
        
        
        
    def score(self,X):
        probs = self.pre




if __name__ == "__main__":
#    X = np.ones([300,3])
#    X[0:100,0]   = 0
#    X[100:200,1] = 0
#    X[200:300,2] = 0
#    
#    bmm = VBBMM(n_components = 3, n_iter = 100, alpha0 = 10, n_init = 5,
#                compute_score = True, c = 1, d = 1, verbose = False)
#    bmm.fit(csr_matrix(X))
#    cluster = bmm.predict(X)
#    resps   = bmm.predict_proba(X)
#    scores  = bmm.score(X)
#    print cluster
#    print "\n"
    #print np.exp(scores)
#    
#    # -------  Example with kaggle digit dataset ----------
#    import pandas as pd
#    import time
#    
#    Data = pd.read_csv('train.csv')
#    data = Data[Data['label']<=10]
#    X    = np.array(data[data.columns[1:]])
#    X[(X>0)*(X<50)] = 25
#    X[(X >= 50)*(X < 150)] = 100
#    X[(X >= 150)*(X < 250)] = 200
#    X[(X >= 250)] = 255
    
#    t1 = time.time()
#    x = csr_matrix(X)
#    t2 = time.time()
#    print t2-t1    
#    X[X>0] = 10
#    #x    = csr_matrix(X)
#    x = csr_matrix(X)
#    bmm = VBBMM(n_components = 10, n_iter = 100, alpha0 = 10, n_init = 2,
#                compute_score = False, c = 1, d = 1, verbose = True)
#    t1 = time.time()
#    bmm.fit(x)
#    t2 = time.time()
#    print t2-t1
#
####################
    
#    X = np.array([ [1,2,3],
#                   [1,2,3],
#                   [1,2,3],
#                   [3,2,1],
#                   [3,2,1],
#                   [3,2,1] ])
##               
#    mmm = VBMMM(n_components = 2, n_iter = 100, alpha0 = 100, compute_score = True,
#                verbose = True, beta0 = 1)
#    mmm.fit(X)
#    
#    def cluster_prototype(obj):
#        prototypes = [0]*obj.n_components
#        for k in range(obj.n_components):
#            prototypes[k] = obj.classes_[np.argmax(obj.means_[k],1)]
#        return prototypes
# 
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    
    def plotter(X, max_k,title, rand_state = 1, prune_thresh = 0.02, mfa_max_iter = 10):
        '''
        Plotting function for VBGMMARD clustering
        
        Parameters:
        -----------
        X: numpy array of size [n_samples, n_features]
           Data matrix
           
        max_k: int
           Maximum number of components
           
        title: str
           Title of the plot
           
        Returns:
        --------
        :instance of VBGMMARD class 
        '''
        # fit model & get parameters
        gmm = VBGMMARD(n_iter = 150, n_components = max_k, n_mfa_iter = mfa_max_iter,
                       prune_thresh = prune_thresh)
        gmm.fit(X)
        centers = gmm.means_
        covars  = gmm.covars_
        k_selected = centers.shape[0]
        
        # plot data
        fig, ax = plt.subplots(figsize = (10,6))
        ax.plot(X[:,0],X[:,1],'bo', label = 'data')
        ax.plot(centers[:,0],centers[:,1],'rD', markersize = 8, label = 'means')
        for i in range(k_selected):
            plot_cov_ellipse(pos = centers[i,:], cov = covars[i], ax = ax)
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend(loc = 2)
        plt.title((title + ', {0} initial clusters, {1} selected clusters').format(max_k,k_selected))
        plt.show()
        return gmm
        
        
    # plot_cov_ellipse function is taken from  
    # https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
    
    def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
        """
        Plots an `nstd` sigma error ellipse based on the specified covariance
        matrix (`cov`). Additional keyword arguments are passed on to the 
        ellipse patch artist.
    
        Parameters
        ----------
            cov : The 2x2 covariance matrix to base the ellipse on
            pos : The location of the center of the ellipse. Expects a 2-element
                sequence of [x0, y0].
            nstd : The radius of the ellipse in numbers of standard deviations.
                Defaults to 2 standard deviations.
            ax : The axis that the ellipse will be plotted on. Defaults to the 
                current axis.
            Additional keyword arguments are pass on to the ellipse patch.
    
        Returns
        -------
            A matplotlib ellipse artist
        """
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]
    
        if ax is None:
            ax = plt.gca()
    
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    
        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, fill = False,
                        edgecolor = 'k',linewidth = 4,**kwargs)
    
        ax.add_artist(ellip)
        return ellip
    
    
    X = np.zeros([600,2])
    X[0:200,:]   = np.random.multivariate_normal(mean = (0,7), cov = [[1,0],[0,1]], size = 200)
    X[200:400,:] = np.random.multivariate_normal(mean = (0,0) , cov = [[1,0],[0,1]], size = 200)
    X[400:600,:] = np.random.multivariate_normal(mean = (0,-7) , cov = [[1,0],[0,1]], size = 200)
    
    sy_gmm_1 = plotter(X,20,'Synthetic Example')
    #gmm = VBGMMARD(n_components = 3, n_iter = 10, verbose = True)
    #gmm.fit(X)