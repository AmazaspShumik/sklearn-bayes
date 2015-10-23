import numpy as np



class VRVM(object):
    '''
    Superclass for Variational Relevance Vector Regression and Variational
    Relevance Vector Classification
    '''
    def __init__(self,max_iter_approx = 20, conv_thresh_approx = 1e-3, max_iter_fit    = 10,
                                                                       conv_thresh_fit = 1e-3,
                                                                       bias_term       = True, 
                                                                       prune_thresh    = 1e-3):
        self.max_iter_approx     = max_iter_approx
        self.conv_thresh_approx  = conv_thresh_approx
        self.max_iter_fit        = max_iter_fit
        self.conv_thresh_fit     = conv_thresh_fit
        self.bias_term           = bias_term
        self.prune_thresh        = prune_thresh
        
        
        
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
        
    '''
    
    def __init__(self,X,Y,**kwargs):
        
        # call constructor of superclass 
        super(self,VRVR).__init__(kwargs)
        
        # number of features & dimensionality
        self.n, self.m   = X.shape
        
        # add bias term if required
        if self.bias_term is True:
            bias         = np.ones([self.n,1])
            self.X       = np.concatenate((bias,X), axis = 1)
            self.m      += 1
        self.Y           = Y
        
          
    def fit(self):
        
       
         
    def predict(self):
        pass
       
         
    def predict_dist(self):
        pass
        
        
        
        
class VRVC(VRVM):
    '''
    Variational Relevance Vector Classifier
    '''
    pass
        
        
    
    
