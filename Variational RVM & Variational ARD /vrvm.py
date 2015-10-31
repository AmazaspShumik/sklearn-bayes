import numpy as np
import scipy as sp
from sklearn.linear_model import LinearRegression



class VRVM(object):
    '''
    Superclass for Variational Relevance Vector Regression and Variational
    Relevance Vector Classification
    '''
    def __init__(self, X, Y, a = None, b = None, kernel = 'rbf', scaler = 1, order              = 2, 
                                                                             max_iter           = 20,
                                                                             conv_thresh        = 1e-3,
                                                                             bias_term          = True, 
                                                                             prune_thresh       = 1e-3):
        self.max_iter            = max_iter
        self.conv_thresh         = conv_thresh
        self.bias_term           = bias_term
        self.prune_thresh        = prune_thresh
        
        # kernelise data if asked (if not kernelised, this is equivalent to
        # ARD regression / classification)
        if kernel is None:
            self.X               = X
        else:
            # check that kernels are supported 
            assert kernel in ['poly','hpoly','rbf','cauchy'],'kernel provided is not supported'
            self.X               = self._kernelise(X,X,kernel,scaler,order)
        
        # number of features & dimensionality
        self.n, self.m           = X.shape
        
        # add bias term if required
        if self.bias_term is True:
            bias                 = np.ones([self.n,1])
            self.X               = np.concatenate((bias,self.X), axis = 1)
            self.m              += 1
        self.Y                   = Y
        
        # number of features used 
        self.active              = np.array([True for i in xrange(self.m)])
        
        # parameters of Gamma distribution for weights
        if a is None:
            self.a = 1e-6*np.ones(self.m) # constant in Bishop & Tipping (2000)
        else:
            assert a.shape[0] == self.m, 'incorrect number of weight parameters'
            self.a = a 
        if b is None:
            self.b = 1e-6*np.ones(self.m) # constant in Bishop & Tipping (2000)
        else:
            assert b.shape[0] == self.m, 'incorrect number of weight parameters'
            self.b = b
            
        # randomly initialise weights
        
        # list of lower bounds (list is updated at each iteration of Mean Field Approximation)
        self.lower_bound = [np.NINF]
    
    
    def _check_convergence(self):
        '''
        Checks convergence of lower bound
        
        Returns:
        --------
        : bool
          If True algorithm converged, if False did not.
            
        '''
        lb = self._lower_bound()
        self.lower_bound.append(lb)
        if self.lower_bound[-1] - self.lower_bound[-2] < self.conv_thresh:
            return True
        return False
        
                        
    def _lower_bound(self):
        raise NotImplementedError()
            
    
    @staticmethod
    def _kernelise(X,Y, kernel, scaler, p):
        '''
        Transforms features through kernelisation (user can add 
        his own kernels, note kernel do not need to be Mercer kernels).
        
        Parameters:
        -----------
        X1: numpy array of size [n_samples_1, n_features]
           First design matrix
           
        X2: numpy array of size [n_samples_2, n_features]
           Second design matrix
           
        kernel: str (DEFAULT = 'rbf')
           Kernel type (currently only 'poly','hpoly','cauchy','rbf' are supported)
           
        scaler: float (DEFAULT = 1)
           Scaling constant for polynomial & rbf kernel
           
        p: int (DEFAULT = 1 )
           Order of polynomial (applied only to polynomial kernels)
           
           
        Returns:
        --------
        K: numpy array of size [n_samples, n_samples]
           Kernelised feature matrix
           
        '''
        # precompute (used for all kernels)
        XY = np.dot(X,Y.T)
        
        # squared distance
        def distSq(X,Y,XY):
            ''' Calculates squared distance'''
            return (-2*XY + np.sum(Y*Y, axis=1)).T + np.sum(X*X, axis = 1)
        
        # construct different kernels
        if kernel == "poly":
            # non-stationary polynomial kernel
            return (1 + XY / scaler )**p
        elif kernel == "rbf":
            # Gaussian kernel
            dsq  = distSq(X,Y,XY)
            K    = np.exp( -dsq / scaler)
            return K
        elif kernel == "hpoly":
            # stationary polynomial kernel
            return (XY / scaler)**p
        else:
            # cauchy kernel
            dsq  = distSq(X,Y,XY) / scaler
            return 1. / (1 + dsq)
            
        
class VRVR(VRVM):
    '''
    Variational Relevance Vector Regressor
    
    Parameters:
    -----------
    
    X: numpy array of size [n_samples,n_features]
       Matrix of explanatory variables
       
    Y: numpy array of size [n_samples,1]
       Vector of dependent variable
       
    max_iter_approx: int
       Maximum number of iterations for mean-field approximation
       
    conv_thresh_approx: float
       Convergence threshold for lower bound change
       
    bias_term: bool
       If True will use bias term
       
    prune_thresh: float
       Threshold for pruning out variable
       
    
    
        
    '''
    
    def __init__(self,c = 1, d = 1,*args,**kwargs):
        super(self,VRVR).__init__(*args,**kwargs) 

          
    def fit(self):
        '''
        Fits variational relevance vector regression
        '''
        
        # precompute some values for faster iterations 
        XY = np.dot(self.X.T,self.Y)
        Y2 = np.sum(self.Y**2)
        
        for i in range(self.max_iter):
            
            # update q(w)
            
            
            
            # update q(tau)
            
            
            # update q(alpha_{j}) for j = 1:n_features
            
            # check convergence
            conv = self._check_convergence()
            
        
    
    def predict(self, X):
        '''
        Calculates mean of predictive distribution
        
        Parameters:
        -----------
        X:     numpy array of size [unknown,n_features]
               Matrix of explanatory variables 
        
        Returns:
        --------
        
        y_hat: numpy array of size [unknown,n_features]
               Mean of predictive distribution
                
        '''
        # kernelise data
        x = self._kernelise(X,self.support_vecs)
       
         
    def predict_dist(self, X):
        '''
        Calculates mean and variance of predictive distribution
        
        Parameters:
        -----------
        X:     numpy array of size [unknown,n_features]
               Matrix of explanatory variables 
        
        Returns:
        --------
        [y_hat, var_hat]: list of two numpy arrays
        
        y_hat: numpy array of size [unknown, n_features]
               Mean of predictive distribution
               
        var_hat: numpy array of size [unknown, n_features]
               Variance of predictive distribution for every observation
        '''
        # kernelise data
        x = self._kernelise(X,self.support_vecs)
        
        
    def _posterior_weights(self, X, XY, YY, exp_tau, exp_A, full_covar = False):
        '''
        Calculates parameters of posterior distribution of weights
        
        Parameters:
        -----------
        X:  numpy array of size n_features
            Matrix of active features (changes at each iteration)
        
        XY: numpy array of size [n_features]
            Dot product of X and Y (for faster computations)
            
        YY: float
            Dot product of Y and Y (for faster computation)
            
        exp_tau: float
            Mean of precision parameter of likelihood
            
        exp_A: numpy array of size n_features
            Vector of 
        
        full_covar: bool
           If True returns covariance matrix, if False returns only diagonal ele-
           ments of covariance matrix. This allows faster computation.
           
        Returns:
        --------
        [y_hat, var_hat]: list of two numpy arrays
        
        y_hat: mean of posterior distribution
        var_hat: diagonal of variance matrix or full variance matrix
        '''
        
        # compute precision parameter
        S = 
        
        # cholesky decomposition
        R = np.linalg.cholesky(X)
        
        
        
        
        
        
        
        
        
        
        
class VRVC(VRVM):
    '''
    Variational Relevance Vector Classifier
    '''
    pass
    
if __name__ == "__main__":
    # testing kernel method
    X     = np.random.random([10,2])
    Y     = np.random.random([5,2])
    vrvm     = VRVM(X,Y,kernel = "rbf", scaler = 1, order = 1)
        
        
    
    
