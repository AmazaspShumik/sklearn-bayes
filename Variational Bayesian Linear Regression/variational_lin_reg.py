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
    
    def __init__(self,X,Y, ab0 = None, cd0 = None, bias_term = True, max_iter = 20, 
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
        
        # svd decomposition & precomputed values for speeding up iterations
        self.u,self.D     =  None,None
        self.vt, self.XY  =  None,None
                
        
    def fit(self):
        '''
        Fits Variational Bayesian Linear Regression Model
        '''
        # SVD decomposition, done once , reused at each iteration
        self.u,self.D, self.vt = np.linalg.svd(self.X, full_matrices = False)
        
        # compute X'*Y  &  Y'*Y to reuse in each iteration
        self.XY                = np.dot(self.X.T,self.Y)
        YY                     = np.dot(self.Y,self.Y)
        
        # some parameters of Gamma distribution have closed form solution
        self.a                 = self.a + float(self.m) / 2
        self.c                 = self.c + float(self.n) / 2
        b,d                    = self.b,self.d
        
        for i in range(self.max_iter):
            
            #  ----------   UPDATE Q(beta_)   --------------
            
            # calculate expected values of alpha and lambda
            E_lambda     = self._gamma_mean(self.c,d)
            E_alpha      = self._gamma_mean(self.a,b)
            
            # update parameters of Q(beta_)
            self.beta,Sn  = self._posterior_dist_beta(E_lambda, E_alpha)
            
            #  ----------    UPDATE Q(alpha_)   ------------
            
            # update rate parameter for Gamma distributed precision of weights
            b            = self.b + 0.5*np.dot(self.beta,self.beta) + 0.5*np.trace(Sn)
            
            #  ----------    UPDATE Q(lambda_)   ------------

            # update rate parameter for Gamma distributed precision of likelihood            
            Xbeta        = np.sum(np.dot(self.X,self.beta)**2)
            XSX          = np.trace(np.dot(np.dot(self.X,Sn),self.X.T))
            d            = self.d + 0.5*(YY - 2*np.dot(self.beta,self.XY) + XSX + Xbeta)
            
            # check convergence 
            converged = False
            
            # save fitted parameters
            if converged or i==(self.max_iter-1):
                
                # save parameters of Gamma distribution
                self.b, self.d        = b, d
                
                # compute parameters of weight distributions corresponding
                # to new alpha_ & lambda_
                E_lambda              = self._gamma_mean(self.c,self.d)
                E_alpha               = self._gamma_mean(self.a,self.b)
                self.beta, self.Sigma = self._posterior_dist_beta(E_lambda,E_alpha)
                
                if converged is False:
                    print("Warning!!! Algorithm did not converge")
                    
                return
             
        
        
        
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
        # mean of predictive distribution
        y_hat     = np.dot(x,self.beta)
        # take into account bias term
        y_hat     = y_hat + self.muY
        
        return y_hat
        
        
    def predict_dist(self,X):
        '''
        Computes parameters of predictive distribution
        
        Parameters:
        -----------
        
        X: numpy array of size [unknown, n_features]
           Matrix of explanatory variables for test set
           
        Returns:
        --------
        [y_hat,var]: list of size two
        
        y_hat: numpy array of size [unknown, 1]
           Mean of predictive distribution at each point
           
        var: numpy array of size [unknown, 1]
           Variance of predictive distribution at each point
           
        '''
        # center data
        x         = X - self.muX
        # mean of predictive distribution
        y_hat     = np.dot(x,self.beta)
        # take into account bias term
        y_hat     = y_hat + self.muY
        
        # asymptotic noise
        noise     = 1./ self._gamma_mean(self.c,self.d)
        var       = noise + np.sum(np.dot(x,self.Sigma)*x,axis = 1)
        return [y_hat,var]
        
        
    def _posterior_dist_beta(self, E_lambda, E_alpha):
        '''
        Calculates parameters of approximation of posterior distribution 
        of weights
        
        Parameters:
        -----------
        
        E_lambda: float
           Expectation of likelihood precision parameter with respect to 
           its factored distribution Q(lambda_)
        
        E_alpha: float
           Expectation of precision parameter for weight distribution with
           respect to its factored distribution Q(alpha_)
        
        Returns:
        --------
        : list of two numpy arrays [Mn,Sn]
        
        Mn: numpy array of size [n_features, 1]
           Mean of posterior ditribution of weights
            
        Sn: numpy array of size [n_features, n_features]
           Covariance of posterior distribution of weights 
            
        '''
        # inverse eigenvalues of precision matrix
        Dinv = 1. / (E_lambda*self.D**2 + E_alpha)
        
        # Covariance matrix ( use numpy broadcasting to speed up)
        Sn   = np.dot(self.vt.T*(Dinv),self.vt)
        
        # mean of approximation for posterior distribution
        Mn   = E_lambda * np.dot(Sn,self.XY)
        return [Mn,Sn]
        
        
    
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
