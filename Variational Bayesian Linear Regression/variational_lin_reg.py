import numpy as np



class VariationalLinearRegression(object):
    '''
    Implements fully Bayesian Linear Regression using 
    mean-field approximation over latent variables.
    Assumes gamma prior on precision 
    
    Parameters:
    -----------
    
    X: numpy array of size [n_samples,n_features]
       Matrix of explanatory variables
       
    Y: numpy array of size [n_samples,1]
       Vector of dependent variables
       
    bias_term: bool (DEFAULT = True)
       If True will use bias term, if False bias term is not used
       
    max_iter: int (DEFAULT = 10)
       Maximum number of iterations for KL minimization
       
    conv_thresh: float (DEFAULT = 1e-3)
       Convergence threshold
       
    '''
    
    def __init__(self,X,Y,bias_term = True, max_iter = 10, conv_thresh = 1e-3):
        self.muX   = np.mean() 