import numpy as np
from scipy.optimize import fmin_l_bfgs_b

#---------------------- Helper functions for logistic ------------------------#

def cost_grad(X,Y,w,alpha):
    '''
    Calculates cost and gradient for logistic regression
    
    '''
    Xw    = np.dot(X,w)
    s     = sigmoid(Xw)
    cost  =  - np.log(s)*Y - np.log(1 - s)*(1 - Y) + alpha*np.dot(w,w) / 2
    grad  = np.dot(X.T, Y - s) + alpha*w
    return [cost,grad]




class BayesianLogisticRegression(object):
    '''
    Implements Bayesian Logistic Regression with type II maximum likelihood, uses
    Gaussian (Laplace) method for approximation of evidence function.
    
    Parameters:
    -----------
    X: numpy matrix of size 'n x m'
       Matrix of explanatory variables       
       
    Y: numpy vector of size 'n x 1'
       Vector of dependent variables
    
    max_iter: float 
       Maximum number of iterations
    
    conv_thresh: float 
       convergence threshold
       
    '''
    
    
    def __init__(self,X,Y):
        # all classes in Y
        classes = set(Y)
        # check that there are only two classes in vector Y
        assert len(classes)==2,"Number of classes in dependent variable should be 2"
        # convert dependent variable to 0 1 vector
        
        
    def fit(self):
        '''
        Fits Bayesian Logistic Regression with Laplace approximation 
        '''
        self._evidence_maximize()
        
        
    def predict_prob(self,X):
        '''
        Predicts probability of observation being of particular class
        
        Parameters:
        -----------
        
        X: numpy array of size 'unknown x m'
           Matrix of explanatory variables test data
           
        Returns:
        --------
        
        P: numpy array of size 'unknown x 1'
           Matrix of probabilities
        '''
        
    

    def _irls(self, X, Y, alpha, Theta):
        '''
        Iteratively refitted least squares method using l_bfgs_b.
        Finds MAP estimates for weights and Hessian at convergence point
        
        Returns:
        --------
        
        [Wmap,negHessian]: list of two numpy arrays
              
              Wmap: numpy array of size 'm x 1'
                    Mode of posterior distribution (mean for Gaussian approx.)
                    
              negHessian: numpy array of size 'm x m'
                    Covariance of Gaussian (Laplace) approximation
        '''
        Sigma = sigmoid(np.dot(X,Theta))
        cost = Y*
        grad = 
        

    
    def _evidence_maximize(self):
        '''
        Maximize evidence (type II maximum likelihood)
        '''
        pass
        
        
        

    
    