# -*- coding: utf-8 -*-
import numpy as np
from scipy.special import psi
from scipy.special import gammaln
from sklearn.cluster import KMeans
from scipy.linalg import pinvh
from scipy.misc import logsumexp
from scipy.sparse import csr_matrix
from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator
from sklearn.utils import check_X_y, check_array
from sklearn.utils import check_is_fitted
import warnings


#TODO: lower bound & convergence check using lower bound
#TODO: Use column sparse matrices in multi label classification



#----------  Variational Gaussian Mixture Model with Automatic Relevance Determination ---------#

class VBGMMARD(object):
    '''
    Variational Bayeisian Gaussian Mixture Model with Automatic Relevance 
    Determination. Implemented model automatically selects number of relevant 
    components through mixture of Type II Maximum Likelihood and Mean Field
    Approximation. In constrast to standard 
    
    Parameters:
    -----------       
    max_components: int
       Maximum number of mixture components
       
    means: numpy array of size [max_components,n_features] or None (DEFAULT = None)
       Cluster means for prior distribution
       
    dof: int or None (DEFAULT = None)
       Degrees of freedom for prior distribution

    covar: numpy array of size [n_features,n_features] or None (DEFAULT = None)
       Inverse of scaling matrix for prior wishart distribution
     
    weights: numpy array of size [max_components,1] or None (DEFAULT = None)
       Latent variable distribution parameter (cluster weights)
    
    beta: float (DEFAULT = 1e-3) 
       scaling constant for precision in mean's prior 
       
    max_iter: int (DEFAULT = 10) 
       Maximum number of iterations
       
    conv_thresh: float (DEFAULT = 1e-3) 
       Convergence threshold 
       
    prune_thresh: float
       Threshold for pruning components
       
    n_kmean_inits: int
       Number of time k-means algorithm will be rerun before the best model is selected
       
    rand_state: int
       Random number that is used for initialising centroids (is passed to k-means)
       
    mfa_max_iter: int
       Maximum number of iterations for Mean Field Approximation of lower bound for 
       evidence function 
       
    References:
    -----------
    Adrian Corduneanu and Chris Bishop, Variational Bayesian Model Selection 
    for Mixture Distributions (2001)
    '''
    def __init__(self, max_components,means = None, dof = None, covar = None,  
                       weights = None, beta = 1e-3, max_iter = 100,
                       conv_thresh = 1e-5,n_kmean_inits = 3, prune_thresh = 1e-5,
                       rand_state = 1, mfa_max_iter = 1):
        self.n_components               =  max_components
        self.dof0, self.scale_inv0      =  dof,covar
        self.weights0,self.means0       =  weights,means
        self.beta0                      =  beta
        self.max_iter,self.conv_thresh  =  max_iter, conv_thresh
        self.n_kmean_inits              =  n_kmean_inits
        self.prune_thresh               =  prune_thresh
        self.rand_state                 =  rand_state
        self.mfa_max_iter               =  mfa_max_iter
        self.converged                  =  False
        # parameters of predictive distribution
        self.predictors                 =  None
        # boolean that identifies whther model was fitted or not
        self.is_fitted                  =  True
        
      
    def _init_params(self,X):
        '''
        Initialise parameters
        '''
        self.n, self.d         = X.shape
        
        # Initialise parameters for all priors, these parameters are used in 
        # variational approximation at each iteration so they should be saved
        # and not changed
            
        # initialise prior on means & precision matrices
        if self.means0 is None:
            kms = KMeans(n_init = self.n_kmean_inits, n_clusters = self.n_components, 
                         random_state = self.rand_state)
            self.means0     = kms.fit(X).cluster_centers_
            
        # broad prior over precision matrix
        if self.scale_inv0 is None:
            # heuristics that seems to work pretty good
            diag_els        = np.abs(np.max(X,0) - np.min(X,0))
            self.scale_inv0 = np.diag( diag_els  )
            self.scale0     = np.diag( 1./ diag_els )
            
        # initialise weights
        if self.weights0 is None:
            self.weights0  = np.ones(self.n_components) / self.n_components
          
        # initial number of degrees of freedom
        if self.dof0 is None:
            self.dof0           = self.d
            
        # clusters that are not pruned 
        self.active             = np.array([True for _ in range(self.n_components)])
        
        # checks initialisation errors in case parameters are user defined
        assert self.dof0 >= self.d,( 'Degrees of freedom should be larger than '
                                         'dimensionality of data')
        assert self.means0.shape[0] == self.n_components,('Number of centrods defined should '
                                                          'be equal to number of components')
        assert self.means0.shape[1] == self.d,('Dimensioanlity of means and data '
                                                   'should be the same')
        assert self.weights0.shape[0] == self.n_components,('Number of weights should be equal '
                                                           'to number of components')
        
        # At first iteration these parameters are equal to priors, but they change 
        # at each iteration of mean field approximation
        self.scale   = np.array([np.copy(self.scale0) for _ in range(self.n_components)])
        self.means   = np.copy(self.means0)
        self.weights = np.copy(self.weights0)
        self.dof     = self.dof0*np.ones(self.n_components)
        self.beta    = self.beta0*np.ones(self.n_components)
        

    def _update_logresp_k(self, X, k):
        '''
        Updates responsibilities for single cluster, calculates expectation
        of logdet of precision matrix.
        
        Parameters:
        -----------
        X: numpy array of size [n_samples,n_features] 
           Data matrix
           
        k: int
           Cluster index

        Returns:
        --------
        log_pnk: numpy array of size [n_features,1]
                 Responsibilities without normalisation
        '''
        # calculate expectation of logdet of precision matrix
        scale_logdet   = np.linalg.slogdet(self.scale[k])[1]
        e_logdet_prec  = sum([psi(0.5*(self.dof[k]+1-i)) for i in range(1,self.d+1)])
        e_logdet_prec += scale_logdet + self.d*np.log(2)
           
        # calculate expectation of quadratic form (x-mean_k)'*precision_k*(x - mean_k)
        x_diff         = X - self.means[k,:]
        e_quad_form    = np.sum( np.dot(x_diff,self.scale[k,:,:])*x_diff, axis = 1 )
        e_quad_form   *= self.dof[k]
        e_quad_form   += self.d / self.beta[k] 
        
        # responsibilities without normalisation
        log_pnk        = np.log(self.weights[k]) + 0.5*e_logdet_prec - 0.5*e_quad_form
        log_pnk       -= self.d * np.log( 2 * np.pi)
        return log_pnk
                
                
    def _update_resps(self,X):
        '''
        Updates distribution of latent variable (responsibilities)
        
        Parameters:
        -----------
        X: numpy array of size [n_samples,n_features] 
           Data matrix

        Returns:
        --------
        p: numpy array of size [n_samples, n_components]
           Responsibilities
        '''
        # log of responsibilities before normalisaton
        log_p     = [self._update_logresp_k(X,k) for k in range(self.n_components)]
        log_p     = np.array(log_p).T
        log_p    -= logsumexp(log_p, axis = 1, keepdims = True)
        p         = np.exp(log_p)
        return p
    
    
    def _update_means_precisions(self, Nk, Xk, Sk):
        '''
        Updates distribution of means and precisions. 
        
        Parameters:
        -----------
        Nk: numpy array of size [n_components,1]
            Sum of responsibilities by component
        
        Xk: list of numpy arrays of length n_components
            Weighted average of observarions, weights are responsibilities
        
        Sk: list of numpy arrays of length n_components
            Weighted outer product of observations, weights are responsibilities 
        '''
        for k in range(self.n_components):
            # update mean and precision for each cluster
            self.beta[k]   = self.beta0 + Nk[k]
            self.means[k]  = (self.beta0*self.means0[k,:] + Xk[k]) / self.beta[k]
            self.dof[k]    = self.dof0 + Nk[k] + 1
            # precision calculation is ugly but prevent overflow & underflow
            self.scale[k,:,:]  = pinvh( self.scale_inv0 + (self.beta0*Sk[k] + Nk[k]*Sk[k] - 
                                 np.outer(Xk[k],Xk[k]) - 
                                 self.beta0*np.outer(self.means0[k,:] - Xk[k],self.means0[k,:])) /
                                 (self.beta0 + Nk[k]) )

                             
    def _check_convergence(self,n_components_before,means_before):
        '''
        Checks convergence

        Parameters:
        -----------
        n_components_before: int 
            Number of components on previous iteration
            
        means_before: numpy array of size [n_components, n_features]
            Cluster means on previous iteration
            
        Returns:
        --------
        :bool 
            If True then converged, otherwise not
        '''
        conv = True
        for mean_before,mean_after in zip(means_before,self.means):
            mean_diff = mean_before - mean_after
            conv  = conv and np.sum(np.abs(mean_diff)) / self.d < self.conv_thresh
        return conv
        
        
    def fit(self, X):
        '''
        Fits Variational Bayesian GMM with ARD, automatically determines number 
        of mixtures
        
        Parameters:
        -----------
        X: numpy array [n_samples,n_features]
           Data matrix
        '''
        # initialise all parameters
        self._init_params(X)
        
        # when fitting new model old parmaters for predictive distribution are 
        # not valid any more
        if self.is_fitted is True : self.St = None
        
        active = np.array([True for _ in range(self.n_components)])        
        for j in range(self.max_iter):
            for i in range(self.mfa_max_iter):
                                
                # STEP 1:   Approximate distribution of latent vatiable, means and 
                #           precisions using Mean Field Approximation method
                
                # calculate responsibilities
                resps = self._update_resps(X)
                
                # precalculate some intermediate statistics
                Nk     = np.sum(resps,axis = 0)
                Xk     = [np.sum(resps[:,k:k+1]*X,0) for k in range(self.n_components)]
                Sk     = [np.dot(resps[:,k]*X.T,X) for k in range(self.n_components)]
                          
                # update distributions of means and precisions
                means_before = np.copy(self.means)
                self._update_means_precisions(Nk,Xk,Sk)
                
                # STEP 2: Maximize lower bound with respect to weights, prune
                #         clusters with small weights & check convergence 
                if i+1 == self.mfa_max_iter:
                    
                    # update weights to maximize lower bound  
                    self.weights      = Nk / self.n
                    
                    # prune all irelevant weights
                    active              = self.weights > self.prune_thresh
                    self.means0         = self.means0[active,:]
                    self.scale          = self.scale[active,:,:]
                    self.weights        = self.weights[active]
                    self.weights       /= np.sum(self.weights)
                    self.dof            = self.dof[active]
                    self.beta           = self.beta[active]
                    n_components_before = self.n_components
                    self.means          = self.means[active,:]
                    self.n_components   = np.sum(active)
                    
                    # check convergence
                    if n_components_before == self.n_components:
                        self.converged  = self._check_convergence(n_components_before,
                                                                  means_before)
                    
                    # if converged postprocess
                    if self.converged == True:
                        self.is_fitted = True
                        return
                        
        warnings.warn( ("Algorithm did not converge!!! Maximum number of iterations "
                        "achieved. Try to change either maximum number of iterations "
                        "or conv_thresh parameters"))
        self.is_fitted  = True
        
        
    def predict_cluster_prob(self,x):
        '''
        Calculates of observation being in particular cluster
        
        Parameters:
        -----------
        x: numpy array of size [n_samples_test_set, n_features]
           Data matrix for test set
           
        Returns:
        --------
        : numpy array of size [n_samples_test_set, n_components]
           Responsibilities for test set
        '''
        return self._update_resps(x)
    
    
    def predict_cluster(self,x):
        '''
        Predicts which cluster generated test data
        
        Parameters:
        -----------
        x: numpy array of size [n_samples_test_set, n_features]
           Data matrix for test set
           
        Returns:
        --------
        : numpy array of size [n_samples_test_set, n_components]
           Responsibilities for test set
        '''
        return np.argmax( self._update_resps(x), 1)
        
        
    def _predict_params(self):
        '''
        Calculates parameters for predictive distribution
        '''
        self.predictors = []
        for k in range(self.n_components):
            df    = self.dof[k] + 1 - self.d
            prec  = self.scale[k,:,:] * self.beta[k] * df / (1 + self.beta[k])
            self.predictors.append(StudentMultivariate(self.means[k,:],prec,
                                                       self.dof[k],self.d))
        
        
    def predictive_pdf(self,x):
        '''
        PDF Predictive distribution
        
        Parameters:
        -----------
        x: numpy array of size [n_samples_test_set,n_features]
           Data matrix for test set
           
        Returns:
        --------
        probs : numpy array of size [n_samples_test_set, 1]
           Value of pdf of predictive distribution at x
        '''
        # check whether model is fitted 
        if self.is_fitted is False:
            raise TypeError('Model is not fitted')
            
        # check whether prediction parameters were calculated before
        if self.St is None:
            # if not calculate predictive parameters
            self._predict_params()
        
        # make prediction
        probs = np.zeros(x.shape[0])
        for k,predictor in enumerate(self.predictors):
            probs += self.weights[k]*predictor.pdf(x)
        return probs


    def get_params(self):
        '''
        Returns dictionary with all learned parameters
        '''
        covars = [1./df * pinvh(sc) for sc,df in zip( self.scale, self.dof)]
        params = {'means': self.means, 'covars': covars,'weights': self.weights}
        return params



class VBGMMARDClassifier(ClassifierMixin,BaseEstimator):
    '''
    Generative classifier, density of each class is approximated with VBGMMARD
    and then 
    
    Parameters:
    -----------       
    max_components: int, optional (DEFAULT = 10)
       Maximum number of mixture components
       
    means: list of numpy arrays of size [max_components,n_features] (DEFAULT = None)
       List of cluster means for each class 
       
    dof: list of ints  (DEFAULT = None)
       Degrees of freedom for prior distribution

    covar: list of numpy arrays of size [n_features,n_features] (DEFAULT = None)
       List of inverse of scaling matrices for each class
     
    weights: list of numpy arrays of size [max_components,1] (DEFAULT = None)
       List of cluster weights for each class
    
    beta: float (DEFAULT = 1e-3) 
       Scaling constant for mean's precision (the same for all classes)
       
    max_iter: int (DEFAULT = 10) 
       Maximum number of iterations
       
    conv_thresh: float (DEFAULT = 1e-3) 
       Convergence threshold 
       
    prune_thresh: float
       Threshold for pruning components
       
    n_kmean_inits: int
       Number of time k-means algorithm will be rerun before the best model is selected
       
    rand_state: int
       Random number that is used for initialising centroids (is passed to k-means)
       
    mfa_max_iter: int
       Maximum number of iterations for Mean Field Approximation of lower bound for 
       evidence function 
    '''
    
    def __init__(self, max_components = 10,means = None, dof = None, covar = None,  
                       weights = None, beta = 1e-3, max_iter = 100,
                       conv_thresh = 1e-5,n_kmean_inits = 3, prune_thresh = 1e-5,
                       rand_state = 1, mfa_max_iter = 5):
                           
        self.n_components               =  max_components
        self.dof, self.covar            =  dof,covar
        self.weights,self.means         =  weights,means
        self.beta                       =  beta
        self.max_iter,self.conv_thresh  =  max_iter, conv_thresh
        self.n_kmean_inits              =  n_kmean_inits
        self.prune_thresh               =  prune_thresh
        self.rand_state                 =  rand_state
        self.mfa_max_iter               =  mfa_max_iter
        # boolean that identifies whether model was fitted or not
        self.is_fitted                  =  True
        
        
    def _init_params(self,y):
        '''
        Initialise parameters
        
        Parameters:
        -----------
        y: numpy array of size 
           Ground truth matrix
        '''        
        # small helper function
        def passer(iterable,index):
            if iterable is not None:
                return iterable[index]
            return iterable
        
        # initialise mixture of gaussians for each class
        self.clfs = []
        for i in range(self.binariser.k):
            self.clfs.append( VBGMMARD(max_components = self.n_components[i],
                                       means          = passer(self.means,i),
                                       dof            = passer(self.dof,i), 
                                       covar          = passer(self.covar,i), 
                                       weights        = passer(self.weights,i),
                                       beta           = self.beta,
                                       max_iter       = self.max_iter,
                                       conv_thresh    = self.conv_thresh,
                                       n_kmean_inits  = self.n_kmean_inits,       
                                       prune_thresh   = self.prune_thresh,       
                                       rand_state     = self.rand_state,       
                                       mfa_max_iter   = self.mfa_max_iter)
                             )
        
        # prior 
        self.prior = np.sum(y, axis = 0) / y.shape[0]
        

    def fit(self,X,Y):
        '''
        Fits classification model
        
        Parameters:
        -----------
        X: numpy array of size [n_samples, n_features]
           Matrix of explanatory variables
           
        Y: numpy array of size [n_samples, 1]
           Vector of dependent variables
        '''
        # binarise
        self.binariser = LabelBinariser(Y)
        y              = self.binariser.convert_vec_to_binary_matrix()
        
        # initialise all parameters
        self._init_params(y)
                
        # fit VBGMMARD for each class
        [clf.fit(X[y[:,cl_idx]==1,:]) for cl_idx,clf in enumerate(self.clfs)]
          
        
    def predict(self,x):
        '''
        Predicts class to which observations belong
        
        Parameters:
        -----------
        x: numpy array of size [n_samples_test, n_features]
           Data matrix for test set
           
        Returns:
        --------
        classes: numpy array of size [n_sample_test, 1]
           Predicted class            
        '''
        check_is_fitted(self,'coef_')
        probs = self.predict_prob(x)
        return self.binariser.convert_prob_matrix_to_vec(probs)
        

    def predict_proba(self,x):
        '''
        Predicts class to which observations belong
        
        Parameters:
        -----------
        x: numpy array of size [n_samples_test, n_features]
           Data matrix for test set
           
        Returns:
        --------
        probs: numpy array of size [n_sample_test, n_classes]
           Matrix of probabilities
        '''
        che
        pr = [clf.predictive_pdf(x)*prior for prior,clf in zip(self.prior,self.clfs)]
        P  = np.array(pr).T
        P  = P / np.sum(P, axis = 1, keepdims = True)
        return P
    
#---------------- Multivariate t distribution (Helper class) ----------------------#
      
    
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
        
# ---------------- Label Binariser ( Helper class ) -------------------------------#
        
        
class LabelBinariser(object):
    '''
    Binarize labels in a one-vs-all fashion.
    Allows easily transform vector of targets for classification to ground truth 
    matrix and easily make inverse transformation.
        
    Parameters:
    ------------ 
    Y: numpy array of size 'n_samples x 1'
       Target variables, vector of classes in classification problem
    '''
    def __init__(self,Y):
        self.Y          = Y
        self.n          = np.shape(Y)[0]
        classes         = set(Y)
        self.k          = len(classes)
        self.direct_mapping  = {}
        self.inverse_mapping = {}
        for i,el in enumerate(classes):
            self.direct_mapping[el] = i
            self.inverse_mapping[i] = el
            
            
    def convert_vec_to_binary_matrix(self, compress = False):
        '''
        Converts vector to ground truth matrix
        
        Parameters:
        -----------
        compress: bool
               If True will use csr_matrix to output compressed matrix
                  
        Returns:
        --------
        Y: numpy array of size 'n x k'
               Ground truth matrix , column number represents class index,
               each row has all zeros and only one 1.  
                
        '''
        Y = np.zeros([self.n,self.k])
        for el,idx in self.direct_mapping.items():
            Y[self.Y==el,idx] = 1
        if compress is True:
            return csr_matrix(Y)
        return Y
            
            
    def convert_binary_matrix_to_vec(self,B, compressed = False):
        '''
        Converts ground truth matrix to vector of classificaion targets
        
        Parameters:
        -----------
        compressed: bool
             If True input is csr_matrix, otherwise B is numpy array
            
        Returns:
        ---------
        Y: numpy array of size 'n x 1'
            Vector of targets, classes
        '''
        if compressed is True:
            B = B.dot(np.eye(np.shape(B)[1]))
        Y = np.zeros(self.n, dtype = self.Y.dtype)
        for i in range(np.shape(B)[1]):
            Y[B[:,i]==1] = self.inverse_mapping[i]
        return Y
        
        
    def convert_prob_matrix_to_vec(self,Y):
        '''
        Converts matrix of probabilities to vector of classification targets
        
        Parameters:
        -----------
        Y:  numpy array of size [n_samples,n_classes]
            Matrix of class probabilities
            
        Returns:
        --------
        Y: numpy array of size 'n x k'
            Ground truth matrix , column number represents class index. 
        '''
        Y_max = np.argmax(Y, axis = 1)
        Y     = np.array([self.inverse_mapping[e] for e in Y_max])
        return Y

    
    

    
    
            
            
            
            

    
    
    