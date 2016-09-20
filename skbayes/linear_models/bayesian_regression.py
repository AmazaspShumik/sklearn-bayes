import numpy as np
from scipy.linalg import svd
import warnings
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.utils import check_X_y



class EBLinearRegression(RegressorMixin,LinearModel):
    '''
    Bayesian Regression with type II maximum likelihood (Empirical Bayes)
    
    Parameters:
    -----------  
    n_iter: int, optional (DEFAULT = 300)
       Maximum number of iterations
         
    tol: float, optional (DEFAULT = 1e-3)
       Threshold for convergence
       
    optimizer: str, optional (DEFAULT = 'fp')
       Method for optimization , either Expectation Maximization or 
       Fixed Point Gull-MacKay {'em','fp'}. Fixed point iterations are
       faster, but can be numerically unstable (especially in case of near perfect fit).
       
    fit_intercept: bool, optional (DEFAULT = True)
       If True includes bias term in model
       
    perfect_fit_tol: float (DEAFAULT = 1e-5)
       Prevents overflow of precision parameters (this is smallest value RSS can have).
       ( !!! Note if using EM instead of fixed-point, try smaller values
       of perfect_fit_tol, for better estimates of variance of predictive distribution )

    alpha: float (DEFAULT = 1)
       Initial value of precision paramter for coefficients ( by default we define 
       very broad distribution )
       
    verbose: bool, optional (Default = False)
       If True at each iteration progress report is printed out
    
    Attributes
    ----------
    coef_  : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)
        
    intercept_: float
        Value of bias term (if fit_intercept is False, then intercept_ = 0)
        
    alpha_ : float
        Estimated precision of coefficients
       
    beta_  : float 
        Estimated precision of noise
        
    eigvals_ : array, shape = (n_features, )
        Eigenvalues of covariance matrix (from posterior distribution of weights)
        
    eigvecs_ : array, shape = (n_features, n_featues)
        Eigenvectors of covariance matrix (from posterior distribution of weights)

    '''
    
    def __init__(self,n_iter = 300, tol = 1e-3, optimizer = 'fp', fit_intercept = True,
                 perfect_fit_tol = 1e-6, alpha = 1, copy_X = True, verbose = False):
        self.n_iter        =  n_iter
        self.tol           =  tol
        if optimizer not in ['em','fp']:
            raise ValueError('Optimizer can be either "em" of "fp" ')
        self.optimizer     =  optimizer 
        self.fit_intercept =  fit_intercept
        self.alpha         =  alpha 
        self.perfect_fit   =  False
        self.copy_X        =  copy_X
        self.verbose       =  verbose
        self.scores_       =  [np.NINF]
        self.perfect_fit_tol = perfect_fit_tol

            
    def fit(self, X, y):
        '''
        Fits Bayesian Linear Regression using Empirical Bayes
        
        Parameters
        ----------
        X: array-like of size [n_samples,n_features]
           Matrix of explanatory variables (should not include bias term)
       
        y: array-like of size [n_features]
           Vector of dependent variables.
           
        Returns
        -------
        object: self
          self
    
        '''
        # preprocess data
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        n_samples, n_features = X.shape
        X, y, X_mean, y_mean, X_std = self._center_data(X, y, self.fit_intercept,
                                                        False, self.copy_X)
        self._x_mean_  = X_mean
        self._y_mean_  = y_mean
        
        #  precision of noise & and coefficients
        alpha   =  self.alpha
        var_y  = np.var(y)
        # check that variance is non zero !!!
        if var_y == 0 :
            beta = 1e-2
        else:
            beta = 1. / np.var(y)

        # to speed all further computations save svd decomposition and reuse it later
        u,d,vt   = svd(X, full_matrices = False)
        Uy      = np.dot(u.T,y)
        dsq     = d**2
        mu      = 0
    
        for i in range(self.n_iter):
            
            # find mean for posterior of w ( for EM this is E-step)
            mu_old  =  mu
            if n_samples > n_features:
                 mu =  vt.T *  d/(dsq+alpha/beta) 
            else:
                 # clever use of SVD here , faster for large n_features
                 mu =  u * 1./(dsq + alpha/beta)
                 mu =  np.dot(X.T,mu)
            mu =  np.dot(mu,Uy)

            # precompute errors, since both methods use it in estimation
            error   = y - np.dot(X,mu)
            sqdErr  = np.sum(error**2)
            
            if sqdErr / n_samples < self.perfect_fit_tol:
                self.perfect_fit = True
                warnings.warn( ('Almost perfect fit!!! Estimated values of variance '
                                'for predictive distribution are computed using only RSS'))
                break
            
            if self.optimizer == "fp":           
                gamma      =  np.sum(beta*dsq/(beta*dsq + alpha))
                # use updated mu and gamma parameters to update alpha and beta
                # !!! made computation numerically stable for perfect fit case
                alpha      =   gamma  / (np.sum(mu**2) + np.finfo(np.float32).eps )
                beta       =  ( n_samples - gamma ) / (sqdErr + np.finfo(np.float32).eps )
            else:             
                # M-step, update parameters alpha and beta to maximize ML TYPE II
                eigvals    = 1/(beta * dsq + alpha)
                alpha      = n_features / ( np.sum(mu**2) + np.sum(1/eigvals) )
                beta       = n_samples / ( sqdErr + np.sum(dsq/eigvals) )

            # if converged or exceeded maximum number of iterations => terminate
            converged = self._check_convergence(mu_old,mu)
            if self.verbose:
                print( "Iteration {0} completed".format(i) )
                if converged is True:
                    print("Algorithm converged after {0} iterations".format(i))
            if converged or i==self.n_iter -1:
                break
        eigvals     = 1./(beta * dsq + alpha)
        self.coef_  = beta*np.dot(vt.T*d*eigvals ,Uy)
        self._set_intercept(X_mean,y_mean,X_std)
        self.beta_  = beta
        self.alpha_ = alpha
        self.eigvals_ = eigvals
        self.eigvecs_ = vt.T
        return self
            

    def predict_dist(self,X):
        '''
        Calculates  mean and variance of predictive distribution for each data 
        point of test set.(Note predictive distribution for each data point is 
        Gaussian, therefore it is uniquely determined by mean and variance)                    
                    
        Parameters
        ----------
        x: array-like of size (n_test_samples, n_features)
            Set of features for which corresponding responses should be predicted

        Returns
        -------
        :list of two numpy arrays [mu_pred, var_pred]
        
            mu_pred : numpy array of size (n_test_samples,)
                      Mean of predictive distribution
                      
            var_pred: numpy array of size (n_test_samples,)
                      Variance of predictive distribution        
        '''
        mu_pred     = self._decision_function(X)
        data_noise  = 1./self.beta_
        model_noise = np.sum(np.dot(X,self.eigvecs_)**2 * self.eigvals_,1)
        var_pred    =  data_noise + model_noise
        return [mu_pred,var_pred]


    def _check_convergence(self, mu_old, mu):
        '''
        Checks convergence using mean of posterior distribution
        '''
        return np.sum(abs(mu-mu_old)>self.tol) == 0
        
