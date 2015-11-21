import numpy as np



class VBPCA(object):
    '''
    Variational Bayesian Principal Component Analysis.
    
    Automatically identifies 'real' dimensionality, uses Wishart prior for 
    precision matrix
    
    Theoretical Note:
    -----------------
    
    
    
    Parameters:
    -----------
    X: numpy array of size [n,n_features]
       Design Matrix
       
    a: numpy array of size [n_features,1]
       Rate parameters for Gamma distributed 
       
    b: numpy array of size [n_features,1]
       Shape parameter
       
    max_iter: int
       Maximum number of iterations for MFA to converge
       
    conv_thresh: float
       Convergence threshold 
       
    '''
    
    def __init__(self,X,max_iter = 50, conv_thresh = 1e-2):
        
        self.max_iter    = max_iter
        self.conv_thresh = conv_thresh
        
    
    def fit(self,X):
        
        for i in range(self.max_iter):
            
            # single iteration of Mean Field Approximation
            Qw     = 
            Qalpha = 
            Qtau   = 
            Qmu    = 
        
    def transform(self):
        pass
        
    def _lower_bound(self):
        pass
        
    def check_convergence(self):
        pass
        
        