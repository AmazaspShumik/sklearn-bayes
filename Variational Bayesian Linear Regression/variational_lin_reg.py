import numpy as np
from scipy.special import psi
from scipy.special import gammaln



class VariationalLinearRegression(object):
    '''
    Implements fully Bayesian Linear Regression using mean-field approximation 
    over latent variables. Assumes gamma prior on precision of weight distribution
    and likelihood.
    
    Graphical Model Composition:
    -----------------
    
    P ( Y | X, beta_, lambda_) = N( Y | X*beta_, lambda_^(-1)*I)
    P ( beta_ | alpha_ )       = N( beta_ | 0, alpha_^(-1)*I)
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
       
    verbose: bool
       If True at each iteration progress report is printed out
    '''
    
    def __init__(self,X,Y, ab0 = [1e-6,1e-6], cd0 = [1e-6,1e-6], bias_term = True, max_iter = 50, 
                 conv_thresh = 1e-3, verbose = False):
        self.verbose          =  verbose
        self.bias_term        =  bias_term
        if bias_term is True:
            self.muX          =  np.mean(X, axis = 0)
            self.muY          =  np.mean(Y)
            self.X            =  X - self.muX
            self.Y            =  Y - self.muY
        else:
            self.X            =  X
            self.Y            =  Y
        
        # Number of samples & dimensioality of training set
        self.n,self.m      =  self.X.shape
        
        # Gamma distribution params. for precision of weights (beta) & likelihood 
        self.a,self.b      =  ab0
        self.c,self.d      =  cd0
        
        # lower bound (should be non-decreasing)
        self.lower_bound   =  [np.NINF]
        
        # termination conditions for mean-field approximation
        self.max_iter      =  max_iter
        self.conv_thresh   =  conv_thresh

        # covariance of posterior distribution
        self.Mw,self.Sigma =  0,0
        
        # svd decomposition & precomputed values for speeding up iterations
        self.u,self.D      =  0,0
        self.vt, self.XY   =  0,0
        
                
        
    def fit(self):
        '''
        Fits Variational Bayesian Linear Regression Model
        '''
        # SVD decomposition, done once , reused at each iteration
        self.u,self.D, self.vt = np.linalg.svd(self.X, full_matrices = False)
        
        # compute X'*Y  &  Y'*Y to reuse in each iteration
        XY                     = np.dot(self.X.T,self.Y)
        YY                     = np.dot(self.Y,self.Y)
        
        # some parameters of Gamma distribution have closed form solution
        self.a                 = self.a + 0.5*self.m
        self.c                 = self.c + 0.5*self.n
        b,d                    = self.b,self.d
        
        # initial mean of posterior for coefficients
        Mw = 0
                
        for i in range(self.max_iter):
            
            #  ----------   UPDATE Q(w)   --------------
            
            # calculate expected values of alpha and lambda
            e_tau         = self._gamma_mean(self.c,d)
            e_alpha       = self._gamma_mean(self.a,b)
            
            # update parameters of Q(w)
            Mw_old       = np.copy(Mw)
            Mw,Sigma     = self._posterior_dist_beta(e_tau, e_alpha,XY)
            
            #  ----------    UPDATE Q(alpha_)   ------------
            
            # update rate parameter for Gamma distributed precision of weights
            E_w_sq        = ( np.dot(Mw,Mw) + np.trace(Sigma) )
            b             = self.b + 0.5*E_w_sq
            
            #  ----------    UPDATE Q(lambda_)   ------------

            # update rate parameter for Gamma distributed precision of likelihood 
            # precalculate some values for reuse in lower bound calculation           
            XMw           = np.sum(np.dot(self.X,Mw)**2)
            XSX           = np.sum(np.dot(self.X,Sigma)*self.X)
            MwXY          = np.dot(Mw,XY)
            d             = self.d + 0.5*(YY + XSX + XMw) - MwXY
            
            
            # --------- Convergence Check ---------
            
            if self.verbose is True:
                print "Iteration {0} is completed".format(i)
                
            # check convergence 
            converged = self._check_convergence(Mw,Mw_old)
            
            # save fitted parameters
            if converged or i==(self.max_iter-1):
                if converged is True:
                    if self.verbose is True:
                        print "Mean Field Approximation converged"
                # save parameters of Gamma distribution
                self.b, self.d        = b, d
                # compute parameters of weight distributions corresponding
                self.Mw, self.Sigma   = Mw,Sigma
                
                if converged is False:
                    print("Warning!!! Variational approximation did not converge") 
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
        if self.bias_term is True:
           X         = X - self.muX
        # mean of predictive distribution
        y_hat     = np.dot(X,self.Mw)
        # take into account bias term
        if self.bias_term is True:
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
        if self.bias_term is True:
           X         = X - self.muX
        # mean of predictive distribution
        y_hat     = np.dot(X,self.Mw)
        # take into account bias term
        if self.bias_term is True:
           y_hat     = y_hat + self.muY
        
        # asymptotic noise
        noise     = 1./ self._gamma_mean(self.c,self.d)
        var       = noise + np.sum(np.dot(X,self.Sigma)*X,axis = 1)
        return [y_hat,var]
        
        
    def _posterior_dist_beta(self, e_tau, e_alpha,XY):
        '''
        Calculates parameters of approximation of posterior distribution 
        of weights
        '''
        # inverse eigenvalues of precision matrix
        Dinv = 1. / (e_tau*self.D**2 + e_alpha)
        
        # Covariance matrix ( use numpy broadcasting to speed up)
        Sn   = np.dot(self.vt.T*(Dinv),self.vt)
        
        # mean of approximation for posterior distribution
        Mn   = e_tau * np.dot(Sn,XY)
        return [Mn,Sn]
        
        
    def _check_convergence(self,Mw,Mw_old):
        '''
        Checks convergence of Mean Field Approximation
        '''
        if np.sum( abs(Mw - Mw_old) > self.conv_thresh ) == 0:
            return True
        return False
        
        
    @staticmethod
    def _gamma_mean(a,b):
       '''
       Calculates mean of gamma distribution
       '''
       return a/b

    
