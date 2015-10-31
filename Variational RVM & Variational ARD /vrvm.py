import numpy as np
import scipy as sp
from sklearn.linear_model import LinearRegression



class VRVM(object):
    '''
    Superclass for Variational Relevance Vector Regression and Variational
    Relevance Vector Classification
    '''
    def __init__(self, X, Y, kernel = 'rbf', scaler = 1, order = 2, max_iter_approx = 20,
                                                                    conv_thresh_approx = 1e-3,
                                                                    max_iter_fit    = 10,
                                                                    conv_thresh_fit = 1e-3,
                                                                    bias_term       = True, 
                                                                    prune_thresh    = 1e-3):
        self.max_iter_approx     = max_iter_approx
        self.conv_thresh_approx  = conv_thresh_approx
        self.max_iter_fit        = max_iter_fit
        self.conv_thresh_fit     = conv_thresh_fit
        self.bias_term           = bias_term
        self.prune_thresh        = prune_thresh
        
        # kernelise data if asked (if not kernelised, this is equivalent to
        # ARD regression / classification)
        if kernel is None:
            self.X       = X
        else:
            # check that kernels are supported 
            assert kernel in ['poly','hpoly','rbf','cauchy'],'kernel provided is not supported'
            self.X       = self._kernelise(X,X,kernel,scaler,order)
        
        # number of features & dimensionality
        self.n, self.m   = X.shape
        
        # add bias term if required
        if self.bias_term is True:
            bias         = np.ones([self.n,1])
            self.X       = np.concatenate((bias,self.X), axis = 1)
            self.m      += 1
        self.Y           = Y
        
        # number of features used 
        self.active      = np.array([True for i in range self.m])
        
        
    
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
    
    def __init__(self,*args,**kwargs):
        super(self,VRVR).__init__(*args,**kwargs)

          
    def fit(self):
        '''
        Fits variational relevance vector regression
        '''
        pass
        
    
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
        pass
        
        
        
        
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
        
        
    
    
