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
                                                                                   conv_thresh = 1e-5,
                                                                                   verbose = True):
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
        # saved for lower bound calculation
        a_init                 = self.a
        b_init                 = self.b
        c_init                 = self.c
        d_init                 = self.d
        
        # some parameters of Gamma distribution have closed form solution
        self.a                 = self.a + 0.5*self.m
        self.c                 = self.c + 0.5*self.n
        b,d                    = self.b,self.d
        
        for i in range(self.max_iter):
            
            #  ----------   UPDATE Q(w)   --------------
            
            # calculate expected values of alpha and lambda
            e_tau         = self._gamma_mean(self.c,d)
            e_alpha       = self._gamma_mean(self.a,b)
            
            # update parameters of Q(w)
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
            
            
            # --------- Lower Bound and Convergence Check ---------
            
            # lower bound calculation
            self._lower_bound(YY,XMw,MwXY,XSX,Sigma,E_w_sq,a_init,b_init,c_init,d_init,
                                                                                e_tau,
                                                                                e_alpha,
                                                                                b,d)
            if self.verbose is True:
                print "Iteration {0} is completed, lower bound is {1}".format(i,self.lower_bound[-1])
            # check convergence 
            #converged = self._check_convergence()
            converged = False
            
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
        
        Parameters:
        -----------
        
        E_lambda: float
           Expectation of precision parameter of likelihood with respect to 
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
        Dinv = 1. / (e_tau*self.D**2 + e_alpha)
        
        # Covariance matrix ( use numpy broadcasting to speed up)
        Sn   = np.dot(self.vt.T*(Dinv),self.vt)
        
        # mean of approximation for posterior distribution
        Mn   = e_tau * np.dot(Sn,XY)
        return [Mn,Sn]
        
        
    def _check_convergence(self):
        '''
        Checks convergence of Mean Field Approximation
        
        Returns:
        -------
        : bool
          True if approximation converged , False otherwise
        '''
        assert len(self.lower_bound) >=2,'There should be at least 2 values of lower bound'
        if self.lower_bound[-1] - self.lower_bound[-2] < self.conv_thresh:
            return True
        return False
        
        
    def _lower_bound(self,YY,XMw,MwXY,XSX,Sigma,E_w_sq,a_init,b_init,c_init,d_init,e_tau,e_alpha,b,d):
        '''
        Calculates lower bound and writes it to instance variable
        
        Parameters:
        -----------
        YY: float
            Dot product Y.T*Y
            
        XMw: float
             L2 norm of X*Mw, where Mw - mean of posterior of weights
            
        MwXY: float
             Product of posterior mean of weights (Mw) and X.T*Y
             
        XSX: float
             Trace of matrix X*Sigma*X.T, where Sigma - covariance of posterior of weights
             
        Sigma: numpy array of size [self.m,self.m]
             Covariance matrix for Qw(w)
             
        E_w_sq: numpy array of size [self.m , 1]
             Vector of weight squares
            
        a_init: float
           Initial shape parameter for Gamma distributed weights
           
        b_init: float
           Initial rate parameter
           
        c_init: float
           Initial shape parameter for Gamma distributed precision of likelihood
           
        d_init: float
           Initial rate parameter
        
        e_tau: float
             Mean of precision for noise parameter
             
        e_alpha: float
             Vector of means of precision parameters for weight distribution
        
        b: float
           Learned rate parameter of Gamma distribution
        
        d: float
           Learned rate parameter of Gamma distribution
        '''
        # < log(tau) > & < log(alpha) >
        e_log_tau   = psi(self.c) - np.log(d)
        e_log_alpha = psi(self.a) - np.log(b)
        # < log P(Y|Xw, tau^-1) >
        like        = 0.5*self.n*e_log_tau - 0.5*e_tau*(YY + XMw + XSX) - MwXY
        # < log P(w| alpha) >
        weights     = 0.5*self.m*e_log_alpha - 0.5*E_w_sq
        # < log P(alpha) >
        alpha_prior = (a_init - 1)*e_log_alpha - e_alpha*b_init
        # < log P(tau) > 
        tau_prior   = (c_init - 1)*e_log_tau - e_tau*c_init
        # < log q(w) >
        q_w         = -0.5*np.linalg.slogdet(Sigma)[1]
        # < log q(alpha)>
        q_alpha     = self.a*np.log(b) + (self.a - 1)*e_log_alpha - e_alpha*b - gammaln(self.a)
        # < log q(tau)>
        q_tau       = self.c*np.log(d) + (self.c - 1)*e_log_tau - e_tau*d - gammaln(self.c)
        # lower bound calculation
        L           = like + weights + tau_prior + alpha_prior - q_w - q_alpha - q_tau
        self.lower_bound.append(L)
        
        
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
       
       
       
if __name__=='__main__':
    X = np.array([[ 0.1,  -0.1,  -0.2,   0.02],
                  [ 0.3,  -0.3,  -0.6,   0.06],
               [ 0.4,  -0.4,  -0.8,   0.08],
               [ 0.5,  -0.5,  -1.,    0.1 ]])

    Y = np.array([ 0.2,  0.6,  0.8,  1. ])  
    vr = VariationalLinearRegression(X,Y)
    vr.fit()
    y_hat = vr.predict(X)
    print y_hat
 
    
