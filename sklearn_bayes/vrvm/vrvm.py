import numpy as np
from scipy.linalg import solve_triangular
from sklearn.linear_model.base import LinearModel, LinearClassifierMixin
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.utils import check_X_y,check_array
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
from scipy.linalg import solve_triangular
import warnings


class VariationalRegressionARD(LinearModel,RegressorMixin):
    '''
    n_iter: int, optional (DEFAULT = 100)
        Maximum number of iterations

    fit_intercept : boolean, optional (DEFAULT = True)
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered)
        
    tol: float, optional (DEFAULT = 1e-3)
        If absolute change in precision parameter for weights is below threshold
        algorithm terminates.
        
    copy_X : boolean, optional (DEFAULT = True)
        If True, X will be copied; else, it may be overwritten.
        
    verbose : boolean, optional (DEFAULT = True)
        Verbose mode when fitting the model 
       
    a: float, optional, (DEFAULT = 1e-6)
       Shape parameters for Gamma distributed precision of weights
       
    b: float, optional, (DEFAULT = 1e-6)
       Rate parameter for Gamma distributed precision of weights
    
    c: float, optional, (DEFAULT = 1e-6)
       Shape parameter for Gamma distributed precision of noise
    
    d: float, optional, (DEFAULT = 1e-6)
       Rate parameter for Gamma distributed precision of noise
       
    prune_thresh: float
       Threshold for pruning out variable    
    
    
    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)
        
    alpha_ : float
       estimated precisions of noise
       
    active_ : array, dtype = np.bool, shape = (n_features)
       True for non-zero coefficients, False otherwise

    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients
    
       
    Reference:
    ----------
    Bishop & Tipping (2000), Variational Relevance Vector Machines
    Jan Drugowitch (2014), Variational Bayesian Inference for Bayesian Linear 
                           and Logistic Regression
    '''
    def __init__(self,  n_iter = 100, tol = 1e-3, prune_thresh = 1e-3, 
                 fit_intercept = True, normalize = False, a = 1e-6, b = 1e-6, 
                 c = 1e-6, d = 1e-6, copy_X = True,verbose = False):
        self.n_iter = n_iter
        self.tol    = tol
        self.fit_intercept   = fit_intercept
        self.a,self.b        = a,b
        self.c,self.d        = c,d
        self.normalize       = normalize
        self.copy_X          = copy_X
        self.verbose         = verbose
        self.prune_thresh    = prune_thresh
        
        
    def fit(self,X,y):
        '''
        Fits variational relevance ARD regression
                
        Parameters
        -----------
        X: array-like of size [n_samples, n_features]
           Training data, matrix of explanatory variables
        
        y: array-like of size [n_samples, n_features] 
           Target values
           
        Returns
        -------
        self : object
            Returns self.
        '''
        # precompute some values for faster iterations 
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        n_samples, n_features = X.shape
        X, y, X_mean, y_mean, X_std = self._center_data(X, y, self.fit_intercept,
                                                        self.normalize, self.copy_X)
        self._x_mean_ = X_mean
        self._y_mean  = y_mean
        self._x_std   = X_std
        self.lower_bound = [np.NINF]
        XX       = np.dot(X.T,X)
        XY       = np.dot(X.T,y)
        Y2       = np.sum(y**2)
        
        # final update for a and c
        a        = (self.a + 1)
        c        = (self.c + 0.5*n_samples)
        # initial values of b,d before mean field approximation
        d        = self.d
        b        = self.b
        
        Mw0      = 0 
        
        for i in range(self.n_iter):
            
            # -------------  update q(w) ------------
            
            # calculate expectations for precision of noise & precision of weights
            e_tau   = self._gamma_mean(c,d)
            e_A     = self._gamma_mean(a,b)    
                 
            # parameters of updated posterior distribution
            Mw,Ri  = self._posterior_weights(XX,XY,e_tau,e_A)
                
            # ------------ update q(tau) ------------
            
            # update rate parameter for Gamma distributed precision of noise 
            # (note shape parameter does not need to be updated at each iteration)
            
            # XMw, XSX, MwXY are reused in lower bound computation
            XSXd      = np.sum( np.dot(X,Ri.T)**2, axis = 0)
            XMw       = np.sum( np.dot(X,Mw)**2 )    
            XSX       = np.sum( XSXd )
            MwXY      = np.dot(Mw,XY)
            d         = self.d + 0.5*(Y2 + XMw + XSX) - MwXY
            
            # ----- update q(alpha(j)) for each j ----
            
            # update rate parameter for Gamma distributed precision of weights
            # (note shape parameter b is updated before iterations started)
            
            E_w_sq    = Mw**2 + XSXd       # is reused in lower bound 
            b         = self.b + 0.5*E_w_sq
            
            # ---------- check convergence ------------
            
            # print progress report if required
            if self.verbose is True:
               print "Iteration {0} is completed, lower bound equals {1}".format(i,self.lower_bound[-1])
                
            if np.sum( abs(Mw - Mw0) > self.tol) == 0 or i == self.n_iter - 1:
                if self.verbose is True:
                        print "Mean Field Approximation completed"
                break
            Mw0 = Mw
            
                        
        # update parameters after last update
        e_tau       = self._gamma_mean(c,d)
        self.alpha_ = e_tau
        e_A         = self._gamma_mean(a,b)    
        self.coef_, self.sigma_ = self._posterior_weights(XX,XY,e_tau,e_A,True)
        # determine relevant vectors
        self.active_ = np.abs(Mw) > self.prune_thresh
        if np.sum(self.active_) == 0:
            warnings.warn(("Warning!!! All vectors were pruned, choose smaller "
                           "value for parameter prune_thresh, by default this implementation "
                           "will use single rv with largest posterior mean"))
            # choose rv with largest posterior mean
            self.active_[np.argmax(abs(Mw))] = True
        self.coef_[~self.active_] = 0 
        self.sigma_ = self.sigma_[self.active_,:][:,self.active_]
        self._set_intercept(X_mean,y_mean,X_std)
        return self
        
        
        
    def predict_dist(self,X):
        '''
        Computes predictive distribution for test set.
        Predictive distribution for each data point is one dimensional
        Gaussian and therefore is characterised by mean and standard
        deviation.
        
        Parameters
        -----------
        X: {array-like, sparse} [n_samples_test, n_features]
           Test data, matrix of explanatory variables
           
        Returns
        -------
        y_hat: numpy array of size [n_samples_test]
           Estimated values of targets on test set (Mean of predictive distribution)
           
        std_hat: numpy array of size [n_samples_test]
           Error bounds (Standard deviation of predictive distribution)
        '''
        check_is_fitted(self, "coef_")
        x     = (X - self._x_mean_) / self._x_std
        y_hat = np.dot(x,self.coef_) + self._y_mean 
        var_hat   = self.alpha_
        var_hat  += np.sum( np.dot(x[:,self.active_],self.sigma_) * x[:,self.active_], axis = 1)
        std_hat   = np.sqrt(var_hat)
        return y_hat, std_hat
        
    
    
    def _posterior_weights(self, XX, XY, exp_tau, exp_A, full_covar = False):
        '''
        Calculates parameters of posterior distribution of weights
        
        Parameters:
        -----------
        X:  numpy array of size n_features
            Matrix of active features (changes at each iteration)
        
        XY: numpy array of size [n_features]
            Dot product of X and Y (for faster computations)

        exp_tau: float
            Mean of precision parameter of noise
            
        exp_A: numpy array of size n_features
            Vector of precisions for weights
           
        Returns:
        --------
        [Mw, Sigma]: list of two numpy arrays
        
        Mw: mean of posterior distribution
        Sigma: covariance matrix
        '''
        # compute precision parameter
        S    = exp_tau*XX       
        np.fill_diagonal(S, np.diag(S) + exp_A)
        
        # cholesky decomposition
        R    = np.linalg.cholesky(S)
        
        # find mean of posterior distribution
        RtMw = solve_triangular(R, exp_tau*XY, lower = True, check_finite = False)
        Mw   = solve_triangular(R.T, RtMw, lower = False, check_finite = False)
        
        # use cholesky decomposition of S to find inverse ( or diagonal of inverse)
        Ri    = solve_triangular(R, np.eye(R.shape[1]), lower = True, check_finite = False)
        if full_covar:
            Sigma = np.dot(Ri.T,Ri)
            return [Mw,Sigma]
        else:
            return [Mw,Ri]


    @staticmethod
    def _gamma_mean(a,b):
        '''
        Calculates mean of gamma distribution
        '''
        return a / b
        

#TODO
class VariationalClassificationARD(LinearClassifierMixin,BaseEstimator):
    pass

        
        
        
class VRVR(VariationalRegressionARD):
    '''
    Variational Relevance Vector Regression.
    Uses Mean Field Method for approximating fully bayesian regression model. 
    
    Practical Advice:
    -----------------
    For faster convergence & numerical stability of algorithm scale matrix of 
    explanatory variables before fitting model.
    
    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 300)
        Maximum number of iterations

    fit_intercept : boolean, optional (DEFAULT = True)
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered)
        
    tol: float, optional (DEFAULT = 1e-3)
        If absolute change in precision parameter for weights is below threshold
        algorithm terminates.
        
    copy_X : boolean, optional (DEFAULT = True)
        If True, X will be copied; else, it may be overwritten.
        
    verbose : boolean, optional (DEFAULT = True)
        Verbose mode when fitting the model 
        
    kernel: str, optional (DEFAULT = 'poly')
        Type of kernel to be used (all kernels: ['rbf' | 'poly' | 'sigmoid', 'linear']
    
    degree : int, (DEFAULT = 3)
        Degree for poly kernels. Ignored by other kernels.
        
    gamma : float, optional (DEFAULT = 1/n_features)
        Kernel coefficient for rbf and poly kernels, ignored by other kernels
        
    coef0 : float, optional (DEFAULT = 1)
        Independent term in poly and sigmoid kernels, ignored by other kernels
        
    kernel_params : mapping of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as
        callable object, ignored by other kernels
       
    a: float, optional, (DEFAULT = 1e-6)
       Shape parameters for Gamma distributed precision of weights
       
    b: float, optional, (DEFAULT = 1e-6)
       Rate parameter for Gamma distributed precision of weights
    
    c: float, optional, (DEFAULT = 1e-6)
       Shape parameter for Gamma distributed precision of noise
    
    d: float, optional, (DEFAULT = 1e-6)
       Rate parameter for Gamma distributed precision of noise

    prune_proportion: float
       Proportion of features that should be pruned
       
    prune_thresh: float
       Threshold for pruning out variable
        
    Reference:
    ----------
    Bishop & Tipping (2000), Variational Relevance Vector Machines
    Jan Drugowitch (2014), Variational Bayesian Inference for Bayesian Linear 
                           and Logistic Regression
    '''
    
    def __init__(self,  n_iter = 100, tol = 1e-3, prune_thresh = 1e-3, 
                 fit_intercept = True, copy_X = True,verbose = False, kernel = 'poly',
                 degree = 2, gamma  = 1, coef0  = 1, kernel_params = None,
                 a = 1e-6, b = 1e-6, c = 1e-6, d = 1e-6):
        super(VRVR,self).__init__(n_iter, tol, prune_thresh, fit_intercept,
                                  False, a, b,c,d,copy_X,verbose)
        # kernel parameters
        self.kernel = kernel
        self.degree = degree
        self.gamma  = gamma
        self.coef0  = coef0
        self.kernel_params = kernel_params
        
        
    def fit(self,X,y):
        '''
        Fits variational relevance vector regression
                
        Parameters
        -----------
        X: array-like of size [n_samples, n_features]
           Training data, matrix of explanatory variables
        
        y: array-like of size [n_samples, n_features] 
           Target values
           
        Returns
        -------
        self : object
            Returns self.
        '''
        X,y = check_X_y(X,y, dtype = np.float64)
        # kernelise features
        K = self._get_kernel( X, X)
        # use fit method of RegressionARD
        _ = super(VRVR,self).fit(K,y)
        self.relevant_  = np.where(self.active_== True)[0]
        if X.ndim == 1:
            self.relevant_vectors_ = X[self.relevant_]
        else:
            self.relevant_vectors_ = X[self.relevant_,:]
        return self
        
    
    def predict_dist(self,X):
        '''
        Computes predictive distribution for test set.
        Predictive distribution for each data point is one dimensional
        Gaussian and therefore is characterised by mean and standard
        deviation.
        
        Parameters
        -----------
        X: array-like of size [n_samples_test, n_features]
           Test data, matrix of explanatory variables
           
        Returns
        -------
        y_hat: numpy array of size [n_samples_test]
           Estimated values of targets on test set (Mean of predictive distribution)
           
        std_hat: numpy array of size [n_samples_test]
           Error bounds (Standard deviation of predictive distribution)
        '''
        check_is_fitted(self,'coef_')
        X = check_array(X, accept_sparse = None, dtype = np.float64) 
        K = self._get_kernel( X, self.relevant_vectors_)
        K = (K-self._x_mean_[self.active_]) / self._x_std[self.active_]
        y_hat = np.dot(K,self.coef_[self.active_]) + self._y_mean
        var_hat   = self.alpha_
        var_hat  += np.sum( np.dot(K,self.sigma_) * K, axis = 1)
        std_hat   = np.sqrt(var_hat)
        return y_hat, std_hat
        
        
    def _decision_function(self, X ):
        '''
        Computes decision function for regression and classification.
        '''
        check_is_fitted(self,'coef_')
        X = check_array(X, accept_sparse = None, dtype = np.float64) 
        K = self._get_kernel( X, self.relevant_vectors_)
        return np.dot(K,self.coef_[self.active_]) + self.intercept_
    
    
    def _get_kernel(self, X, Y):
        '''
        Calculates kernelised features
        '''
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0  }
        return pairwise_kernels(X, Y, metric=self.kernel, filter_params=True,
                                **params)
