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
       
    ab0: list of floats [a0,b0] (Default = [1e-1, 1e-1])
       Parameters of Gamma prior for variance of likelihood
       
    cd0: list of floats [c0,d0] (Deafult = [1e-1, 1e-1])
       Parameters of Gamma prior for variance of 
       
    bias_term: bool (DEFAULT = True)
       If True will use bias term, if False bias term is not used
       
    max_iter: int (DEFAULT = 10)
       Maximum number of iterations for KL minimization
       
    conv_thresh: float (DEFAULT = 1e-3)
       Convergence threshold
       
    '''
    
    def __init__(self,X,Y,bias_term = True, max_iter = 10, conv_thresh = 1e-3):
        self.muX   =  np.mean(X, axis = 0)
        self.muY   =  np.mean(Y)
        self.X     =  X - self.muX
        self.Y     =  Y - self.muY
        
        
    def fit(self):
        pass
        
    def predict(self,X):
        '''
        Calcuates mean of predictive distribution
        
        Parameters:
        -----------
        
        X: numpy array of size [unknown, n_features]
           Matrix of explanatory variables for test set
           
        Returns:
        --------
        
        y_hat: numpy array of size [unknown, 1]
           Mean of predictive distribution
           
        '''
        # center data
        x         = X - self.muX
        
        # 
        # predict
        #
        y_hat     = None
        
        # take into account bias term
        y_hat     = y_hat + self.muY
        return y_hat
        
        
    def predict_dist(self,X):
        pass
        
        