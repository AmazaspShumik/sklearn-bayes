import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import psi
from scipy.special import gamma
from scipy.special import gammaln



class VRVM(object):
    '''
    Superclass for Variational Relevance Vector Regression and Variational
    Relevance Vector Classification
    '''
    def __init__(self,X,Y,a,b,kernel,scaler,order,max_iter,conv_thresh,bias_term,prune_thresh,
                                                                                 verbose):
        
        self.max_iter            = max_iter
        self.conv_thresh         = conv_thresh
        self.bias_term           = bias_term
        self.prune_thresh        = prune_thresh
        
        # kernel parameters
        self.kernel              = kernel
        self.scaler              = scaler
        self.order               = order

        
        # kernelise data if asked (if not kernelised, this is equivalent to
        # ARD regression / classification)
        self.Xraw                = X
        if self.kernel is None:
            self.X               = X
        else:
            # check that kernels are supported 
            assert kernel in ['poly','hpoly','rbf','cauchy'],'kernel provided is not supported'
            self.X               = self._kernelise(X,X,kernel,scaler,order)
        
        
        # number of features & dimensionality
        self.n, self.m           = self.X.shape
        
        # add bias term if required
        if self.bias_term is True:
            bias                 = np.ones([self.n,1])
            self.X               = np.concatenate((bias,self.X), axis = 1)
            self.m              += 1
        self.Y                   = Y
        
        # number of features used 
        self.active              = np.array([True for i in xrange(self.m)])
        
        # parameters of Gamma distribution for weights
        self.a = a*np.ones(self.m)
        self.b = b*np.ones(self.m)

        # list of lower bounds (list is updated at each iteration of Mean Field Approximation)
        self.lower_bound = [np.NINF]
        
        # print progress report
        self.verbose     = verbose
    
    
    def _check_convergence(self):
        '''
        Checks convergence of lower bound
        
        Returns:
        --------
        : bool
          If True algorithm converged, if False did not.
            
        '''
        assert len(self.lower_bound) >=2, 'need to have at least 2 estimates of lower bound'
        if self.lower_bound[-1] - self.lower_bound[-2] < self.conv_thresh:
            return True
        return False
        
    
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
            return ((-2*XY + np.sum(Y*Y, axis=1)).T + np.sum(X*X, axis = 1)).T
        
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
            
            
#------------------- Variational Relevance Vector Regression ----------------------#          
            
        
class VRVR(VRVM):
    '''
    Variational Relevance Vector Regressor
    
    Parameters:
    -----------
    
    X: numpy array of size [n_samples,n_features]
       Matrix of explanatory variables
       
    Y: numpy array of size [n_samples,1]
       Vector of dependent variable
       
    a: numpy array
       Shape parameters for Gamma distributed precision of weights
       
    b: numpy array
       Rate parameter for Gamma distributed precision of weights
    
    c: numpy array
       Shape parameter for Gamma distributed precision of likelihood
    
    d: numpy array
       Rate parameter for Gamma distributed precision of likelihood
    
    kernel: str
       Kernel type {'rbf','poly','hpoly','cauchy'}
       
    scaler: float
       Scaling constant (applied to all types of kernels)
       
    order: int
       Order of polynomial (applies to kernels {'poly','hpoly'})
    
    max_iter: int
       Maximum number of iterations for mean-field approximation
       
    conv_thresh: float
       Convergence threshold for lower bound change
       
    bias_term: bool
       If True will use bias term
       
    prune_thresh: float
       Threshold for pruning out variable
        
    '''
    
    def __init__(self, X, Y, a = 1e-6, b = 1e-6, c = 1e-6, d = 1e-6, kernel       = 'rbf', 
                                                                     scaler       = 1, 
                                                                     order        = 2, 
                                                                     max_iter     = 20,
                                                                     conv_thresh  = 1e-2,
                                                                     bias_term    = True, 
                                                                     prune_thresh = 1e-2,
                                                                     verbose      = False):
        # call to constructor of superclass
        super(VRVR,self).__init__(X,Y,a,b,kernel,scaler,order,max_iter,conv_thresh,bias_term,
                                                                                   prune_thresh,
                                                                                   verbose)
        
        # parameters of Gamma distribution for precision of likelihood
        self.c   = c
        self.d   = d
        

          
    def fit(self):
        '''
        Fits variational relevance vector regression
        '''
        
        # precompute some values for faster iterations 
        XY       = np.dot(self.X.T,self.Y)
        Y2       = np.sum(self.Y**2)
        
        # initial shape parameters a & c
        a_init   = self.a
        c_init   = self.c 
        
        # final update for a and c
        self.a  += 1
        self.c  += float(self.n)/2
        
        d        = self.d
        b        = self.b
        
        for i in range(self.max_iter):
            
            # -------------  update q(w) ------------
            
            # calculate expectations for precision of noise & precision of weights
            e_tau   = self._gamma_mean(self.c,d)
            e_A     = self._gamma_mean(self.a,b)    
                 
            # parameters of updated posterior distribution
            Mw,Sigma  = self._posterior_weights(XY,e_tau,e_A)
            
            # ------------ update q(tau) ------------
            
            # update rate parameter for Gamma distributed precision of noise 
            # (note shape parameter does not need to be updated at each iteration)
            
            # XMw, XSX, MwXY are reused in lower bound computation
            XMw       = np.sum( np.dot(self.X,Mw)**2 )    
            XSX       = np.sum( np.dot(self.X,Sigma)*self.X )
            MwXY      = np.dot(Mw,XY)
            d         = self.d + 0.5*(Y2 + XMw + XSX) - MwXY
            
            # ----- update q(alpha(j)) for each j ----
            
            # update rate parameter for Gamma distributed precision of weights
            # (note shape parameter b is updated before iterations started)
            E_w_sq    = Mw**2 + np.diag(Sigma)      # is reused in lower bound 
            b         = self.b + 0.5*E_w_sq
            
            # ------ lower bound & convergence -------
            
            # calculate lower bound reusing previously calculated statistics
            self._lower_bound(Y2,XMw,MwXY,XSX,Sigma,E_w_sq,e_tau,e_A,b,d,a_init,c_init) 
            
            # check convergence
            conv = self._check_convergence()
            
            # print progress report if required
            if self.verbose is True:
               print "Iteration {0} is completed, lower bound equals {1}".format(i,self.lower_bound[-1])
                
            if conv is True or i== self.max_iter - 1:
                if self.verbose is True:
                        print "Mean Field Approximation completed"
                    
                # save parameters of Gamma distribution
                self.b, self.d      = b, d
                # save parametres of posterior distribution 
                self.Mw, self.Sigma = Mw, Sigma
                # determine relevant vectors
                self.active         = np.abs(self.Mw) > self.prune_thresh
                # check that there are any relevant vectors at all
                if np.sum(self.active) == 0:
                    raise ValueError('No relevant vectors selected')
                self.rel_vecs       = self.Xraw[self.active[1:],:]
                return 

                
    def predict(self, x):
        '''
        Calculates mean of predictive distribution
        
        Parameters:
        -----------
        X:     numpy array of size [unknown,n_features]
               Matrix of explanatory variables 
        
        Returns:
        --------
         : numpy array of size [unknown,n_features]
               Mean of predictive distribution
                
        '''
        # kernelise data
        if self.kernel is not None:
            x = self._kernelise(x,self.rel_vecs,self.kernel,self.scaler,self.order)
        # bias term
        if self.bias_term is False:
            return np.dot(x,self.Mw[self.active])
        else:
            y_hat  = np.dot(x,self.Mw[1:][self.active[1:]])
            #add bias term if required
            y_hat += self.Mw[0]
            return y_hat
            
            
    def predict_dist(self, x):
        '''
        Calculates mean and variance of predictive distribution
        
        Parameters:
        -----------
        X:     numpy array of size [unknown,n_features]
               Matrix of explanatory variables 
        
        Returns:
        --------
        [y_hat, var_hat]: list of two numpy arrays
        
        y_hat: numpy array of size [unknown, n_features]
               Mean of predictive distribution
               
        var_hat: numpy array of size [unknown, n_features]
               Variance of predictive distribution for every observation
        '''
        # kernelise data if required
        if self.kernel is not None:
           x = self._kernelise(x,self.rel_vecs, self.kernel, self.scaler,self.order)
           
        # add bias term if required
        if self.bias_term is not None:
            n    = x.shape[0] 
            bias = np.ones([n,1],dtype = np.float)
            x    = np.concatenate((bias,x), axis = 1)
            
        y_hat    = np.dot(x,self.Mw[self.active])
        e_tau    = self._gamma_mean(self.c,self.d)
        var_hat  = np.sum(np.dot(x,self.Sigma[self.active,:][:,self.active])*x, axis = 1) + 1./e_tau
        return [y_hat, var_hat]
        
        
    def _posterior_weights(self, XY, exp_tau, exp_A):
        '''
        Calculates parameters of posterior distribution of weights
        
        Parameters:
        -----------
        X:  numpy array of size n_features
            Matrix of active features (changes at each iteration)
        
        XY: numpy array of size [n_features]
            Dot product of X and Y (for faster computations)

        exp_tau: float
            Mean of precision parameter of likelihood
            
        exp_A: numpy array of size n_features
            Vector of precisions for weights
           
        Returns:
        --------
        [Mw, Sigma]: list of two numpy arrays
        
        Mw: mean of posterior distribution
        Sigma: covariance matrix
        '''
        Mw,Sigma = 0,0
        
        # compute precision parameter
        S    = exp_tau*np.dot(self.X.T,self.X)        
        np.fill_diagonal(S, np.diag(S) + exp_A)
        
        # cholesky decomposition
        R    = np.linalg.cholesky(S)
        
        # find mean of posterior distribution
        RtMw = solve_triangular(R, exp_tau*XY, lower = True, check_finite = False)
        Mw   = solve_triangular(R.T, RtMw, lower = False, check_finite = False)
        
        # use cholesky decomposition of S to find inverse ( or diagonal of inverse)
        Ri    = solve_triangular(R, np.eye(self.m), lower = True, check_finite = False)
        Sigma = np.dot(Ri.T,Ri)
        return [Mw,Sigma]
        
        
    def _lower_bound(self,Y2,XMw,MwXY,XSX,Sigma,E_w_sq,e_tau,e_A,b,d,a_init,c_init):
        '''
        Calculates lower bound and writes it to instance variable self.lower_bound.
        Does not include constants that do not change from one iteration to another.
        
        Parameters:
        -----------
        Y2: float
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
        
        e_tau: float
             Mean of precision for noise parameter
             
        e_A: numpy array of size [self.m, 1]
             Vector of means of precision parameters for weight distribution
        
        b: float/int
           Rate parameter of Gamma distribution
        
        d: float/int
           Rate parameter of Gamma distribution
           
        a_init: float
           Initial shape parameter for Gamma distributed weights
           
        c_init: float
           Initial shape parameter for Gamma distributed precision of likelihood
        '''
        # precompute for diffrent parts of lower bound
        e_log_tau       = psi(self.c) - np.log(d)
        e_log_alpha     = psi(self.a) - np.log(b)
        
        # Integration of likelihood Ew[Ealpha[Etau[ log P(Y| X*w, tau^-1)]]]
        like_first      =  0.5 * self.n * e_log_tau
        like_second     =  0.5 * e_tau * (Y2 - 2*MwXY + XMw + XSX)
        like            = like_first - like_second
        
        # Integration of weights Ew[Ealpha[Etau[ log P(w| 0, alpha)]]]
        weights         = 0.5*(np.sum((e_log_alpha)) - np.dot(e_A,E_w_sq))
        
        # Integration of precision parameter for weigts Ew[Ealpha[Etau[ log P(alpha| a, b)]]]
        alpha_prior     = np.dot((a_init-1),e_log_alpha)-np.dot(self.b,e_A)
        
        # Integration of precison parameter for likelihood
        tau_prior       = (c_init - 1)*e_log_tau - e_tau*self.d
        
        # E [ log( q_tau(tau) )]
        q_tau_const     = self.c*np.log(d) - gammaln(self.c)
        q_tau           = q_tau_const - d*e_tau + (self.c-1)*e_log_tau
        
        # E [ log( q_alpha(alpha)]
        q_alpha_const   = np.dot(self.a,np.log(b)) - np.sum(gammaln(self.a))
        q_alpha         = q_alpha_const - np.dot(b,e_A) + np.dot((self.a-1),e_log_alpha)
        
        # E [ log( q_w(w)) ]
        q_w             = -0.5*np.linalg.slogdet(Sigma)[1]
        print -q_w

        # lower bound        
        L = like + weights + alpha_prior + tau_prior - q_w - q_alpha - q_tau
        self.lower_bound.append(L)
        

    @staticmethod
    def _gamma_mean(a,b):
        '''
        Calculates mean of gamma distribution
        '''
        return a / b
        
                     
 
#-------------------- Variational Relevance Vector Classifier --------------------# 
        
              
class VRVC(VRVM):
    '''
    Variational Relevance Vector Classifier
    
    This implementation is equivalent to Variational Bayesian Logistic Regression
    with Automatic Relevance Determination. With kernelised feature matrix this 
    becomes Variational Bayesian Relevance Vector Classifier.
    
    Theoretical Note:
    -----------------
    When using hierarchical prior analytical derivation of Bayeisan Model becomes
    impossible, so we use Jaakkola & Jordan local variational bound that 
    approximates value of lower bound.

    Parameters:
    ----------
    
    X: numpy array of size [n_samples,n_features]
       Matrix of explanatory variables
       
    Y: numpy array of size [n_samples,1]
       Vector of dependent variable
       
    a: numpy array
       Shape parameters for Gamma distributed precision of weights
       
    b: numpy array
       Rate parameter for Gamma distributed precision of weights

    kernel: str
       Kernel type {'rbf','poly','hpoly','cauchy'}
       
    scaler: float
       Scaling constant (applied to all types of kernels)
       
    order: int
       Order of polynomial (applies to kernels {'poly','hpoly'})
    
    max_iter: int
       Maximum number of iterations for mean-field approximation
       
    conv_thresh: float
       Convergence threshold for lower bound change
       
    bias_term: bool
       If True will use bias term
       
    prune_thresh: float
       Threshold for pruning out variable
    '''
    
    
    def __init__(self,X, Y, a = 1e-6, b = 1e-6, kernel = 'rbf', scaler       = 1, 
                                                                      order        = 2, 
                                                                      max_iter     = 20,
                                                                      conv_thresh  = 1e-3,
                                                                      bias_term    = True, 
                                                                      prune_thresh = 1e-2,
                                                                      verbose      = False):
        # call to constructor of superclass
        super(VRVR,self).__init__(X,Y,a,b,kernel,scaler,order,max_iter,conv_thresh,bias_term,
                                                                                   prune_thresh,
                                                                                   verbose)
       
        
        
    def fit(self):
        pass
        
    def _lower_bound(self):
        pass
        
    def _posterior_weights(self):
        pass



if __name__=='__main__':
       import matplotlib.pyplot as plt
       from sklearn.cross_validation import train_test_split
       
       # Linear Model
       
       #X = np.random.random([1000,1])
       #X[:,0] = np.linspace(-2,2,1000)
       #Y = 4*X[:,0]  + 50 + np.random.normal(0,1,1000)
       #X,x,Y,y = train_test_split(X,Y, test_size = 0.3)
       #vrvr = VRVR(X,Y, kernel = None, max_iter = 50)
       #vrvr.fit()
       #y_hat = vrvr.predict(x)
       #plt.plot(x,y,'ro')
       #plt.plot(x,y_hat,'b+')
       #plt.show()
       
       
       # SINC
       X = np.random.random([4000,1])
       X[:,0]  = np.linspace(-8,8,4000)
       Y       = 20*np.sinc(X[:,0]) + np.random.normal(0,1,4000) + 10
       #Y       = 4*X[:,0] + 3 + np.random.normal(0,1,2000)
       X,x,Y,y = train_test_split(X,Y, test_size = 0.3)
       vrvr   = VRVR(X,Y, kernel = 'rbf', max_iter = 200,order = 2, scaler = 1, verbose = True)
       vrvr.fit()
       y_hat,var_hat  = vrvr.predict_dist(x)
       plt.plot(x[:,0],y_hat,'bo')
       plt.plot(x[:,0],y,"r+")
       plt.plot(x[:,0],y_hat + 1.96*np.sqrt(var_hat),"go")
       plt.plot(x[:,0],y_hat - 1.96*np.sqrt(var_hat),"go")
       
       
       # 
       
