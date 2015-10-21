import numpy as np



class VariationalLinearRegression(object):
    '''
    Implements fully Bayesian Linear Regression using mean-field approximation 
    over latent variables. Assumes gamma prior on precision of weight distribution
    and likelihood.
    
    Theoretical Note:
    -----------------
    
    P ( Y | X, beta_, lambda_) = N( Y | X*beta_, lambda_^(-1)*I)
    P ( beta_ | alpha_ )       = N( 
    P ( alpha_ | a, b)         = Ga( alpha_ | a, b)
    P ( lambda_ | c, d)        = Ga( lambda_ | c, d)
    
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
    
    def __init__(self,X,Y, ab0 = None, cd0 = None, bias_term = True, max_iter = 10, 
                                                                     conv_thresh = 1e-3):
        self.muX          =  np.mean(X, axis = 0)
        self.muY          =  np.mean(Y)
        self.X            =  X - self.muX
        self.Y            =  Y - self.muY
        
        # Number of samples & dimensioality of training set
        self.n,self.m     =  self.X.shape
        
        # Gamma distribution params. for precision of weights (beta) & likelihood 
        if ab0 is None:
            self.a,self.b =  [1e-2, 1e-4]
        else:
            self.a,self.b =  ab0
        if cd0 is None:
            self.c,self.d =  [1e-2, 1e-4]
        else:
            self.c,self.d =  cd0
        
        # weights for features of linear regression
        self.beta_        =  np.zeros(self.m)
        
        # lower bound (should be non-decreasing)
        self.lbound       =  []
        
        # termination conditions for mean-field approximation
        self.max_iter     =  max_iter
        self.conv_thresh  =  conv_thresh
        
        # precision paramters of weight distribution & likelihood
        self.lambda_      =  0.
        self.alpha_       =  0.
        
        # covariance of posterior distribution
        self.Sigma        =  np.zeros([self.m, self.m], dtype = np.float)
        
        # svd decomposition matrices
        self.u            =  0
        self.d            =  0
        self.vt           =  0
        
        
    def fit(self):
        '''
        Fits Variational Bayesian Linear Regression Model
        '''
        # SVD decomposition, done once , reused at each iteration
        self.u,self.d, self.vt = np.linalg.svd(self.X, full_matrices = False)
        
        # compute X'*Y  &  Y'*Y to reuse in each iteration
        XY                     = np.dot(self.X,self.Y)
        YY                     = np.dot(self.Y,self.Y)
        
        # some parameters of Gamma distribution have closed form solution
        self.a                 = self.a + float(self.m) / 2
        self.c                 = self.c + float(self.n) / 2
        
        for i in range(self.max_iter):
            
            #  ----------   UPDATE Q(beta_)   --------------
            
            # calculate expected values of alpha and lambda
            E_lambda     = self._gamma_mean(self.c,self.d)
            E_alpha      = self._gamma_mean(self.a,self.b)
            
            # update parameters of Q(beta_)
            self.beta,D  = self._posterior_dist_beta(E_alpha, E_lambda)
            
            #  ----------    UPDATE Q(alpha_)   ------------
            
            # update rate parameter for Gamma distributed precision of weights
            b            = self.b + 0.5*np.dot(self.beta,self.beta) + 0.5*np.sum(D)
            
            #  ----------    UPDATE Q(lambda_)   ------------

            # update rate parameter for Gamma distributed precision of likelihood
            d            = self.d + 0.5*(YY - 2*np.dot(self.beta,XY))
        
        
        
        
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
        
        
    def _posterior_dist_beta(self):
        '''
        Calculates parameters of posterior distribution of weights
        '''
        pass
        
    
    @staticmethod
    def _gamma_mean(a,b):
       '''
       Calculates mean of gamma distribution
    
       Parameters:
       -----------
       a0: float
           Shape parameter
    
       b0: float
           Rate parameters
        
       Returns:
       --------
       : float
           Mean of gamma distribution
       '''
       return a/b
        
        