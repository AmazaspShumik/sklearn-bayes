import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.utils import check_X_y
from scipy.linalg import svd


def _gamma_mean(c,d):
    '''
    Computes mean of gamma distribution
    '''
    return float(c)/d


class VBLinearRegression(RegressorMixin,LinearModel):
    '''
    Implements Bayesian Linear Regression using mean-field approximation.
    Assumes gamma prior on precision parameters of coefficients and noise.

    Parameters:
    -----------
    n_iter: int, optional (DEFAULT = 300)
       Maximum number of iterations for KL minimization

    tol: float, optional (DEFAULT = 1e-3)
       Convergence threshold
       
    fit_intercept: bool, optional (DEFAULT = True)
       If True will use bias term in model fitting

    a: float, optional (Default = 1e-4)
       Shape parameter of Gamma prior for precision of coefficients
       
    b: float, optional (Default = 1e-4)
       Rate parameter of Gamma prior for precision coefficients
       
    c: float, optional (Default = 1e-4)
       Shape parameter of  Gamma prior for precision of noise
       
    d: float, optional (Default = 1e-4)
       Rate parameter of  Gamma prior for precision of noise
       
    verbose: bool, optional (Default = False)
       If True at each iteration progress report is printed out
       
    Attributes
    ----------
    coef_  : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)
        
    intercept_: float
        Value of bias term (if fit_intercept is False, then intercept_ = 0)

    sigma_ : array, shape = (n_features, n_features)
        Estimated covariance matrix of the coefficients
    '''
    
    def __init__(self, n_iter = 300, tol =1e-4, fit_intercept = True, 
                 a = 1e-4, b = 1e-4, c = 1e-4, d = 1e-4, copy_X = True,
                 verbose = False):
        self.n_iter     =  n_iter
        self.tol        =  tol
        self.a,self.b   =  a, b
        self.c,self.d   =  c, d
        self.copy_X     =  copy_X
        self.fit_intercept = fit_intercept
        self.verbose       = verbose
        self.scores_       = [np.NINF]

        
    def fit(self,X,y):
        '''
        Fits Variational Bayesian Linear Regression Model
        
        Parameters
        ----------
        X: array-like of size [n_samples,n_features]
           Matrix of explanatory variables (should not include bias term)
       
        Y: array-like of size [n_features]
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
                                                        False,self.copy_X)
        self._x_mean_  = X_mean
        self._y_mean_  = y_mean
        
        # SVD decomposition, done once , reused at each iteration
        u,D,vt = svd(X, full_matrices = False)
        dsq    = D**2
        UY             = np.dot(u.T,y)
        
        # some parameters of Gamma distribution have closed form solution
        a              = self.a + 0.5 * n_features
        c              = self.c + 0.5 * n_samples
        b,d            = self.b,  self.d
        
        # initial mean of posterior for coefficients
        mu = 0
                
        for i in range(self.n_iter):
            
            # update parameters of distribution Q(weights)
            e_beta       = _gamma_mean(c,d)
            e_alpha      = _gamma_mean(a,b)
            mu_old       = np.copy(mu)
            mu,eigvals   = self._posterior_weights(e_beta, e_alpha,UY,dsq,u,vt,D,X)
            
            # update parameters of distribution Q(precision of weights) 
            b            = self.b + 0.5*( np.sum(mu**2) + np.sum(eigvals))
            
            # update parameters of distribution Q(precision of likelihood)
            sqderr       = np.sum((y - np.dot(X,mu))**2)
            xsx          = np.sum(dsq*eigvals)
            d            = self.d + 0.5*(sqderr + xsx)
 
            # check convergence 
            converged = self._check_convergence(mu,mu_old)
            if self.verbose is True:
                print "Iteration {0} is completed".format(i)
                if converged is True:
                    print("Algorithm converged after {0} iterations".format(i))
               
            # terminate if convergence or maximum number of iterations are achieved
            if converged or i==(self.n_iter-1):
                break
            
        # save necessary parameters    
        e_beta   = _gamma_mean(c,d)
        e_alpha  = _gamma_mean(a,b)
        self.coef_, self.eigvals_ = self._posterior_weights(e_beta, e_alpha, UY,
                                                            dsq, u, vt, D, X)
        self._set_intercept(X_mean,y_mean,X_std)
        self.eigvecs_ = vt.T
        self._c_ , self._d_ = c,d
        return self
        
        
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
        mu_pred     = self._decision_function(X)
        data_noise  = 1./_gamma_mean(self._c_, self._d_)
        model_noise = np.sum(np.dot(X,self.eigvecs_)**2 * self.eigvals_,1)
        var_pred    =  data_noise + model_noise
        return [mu_pred,var_pred]
        
        
    def _posterior_weights(self, e_beta, e_alpha, UY, dsq, u, vt, d, X):
        '''
        Calculates parameters of approximate posterior distribution 
        of weights
        '''
        # eigenvalues of covariance matrix
        sigma = 1./ (e_beta*dsq + e_alpha)
        
        # mean of approximate posterior distribution
        n_samples, n_features = X.shape
        if n_samples > n_features:
             mu =  vt.T *  d/(dsq+e_alpha/e_beta) 
        else:
             mu =  u * 1./(dsq + e_alpha/e_beta)
             mu =  np.dot(X.T,mu)
        mu =  np.dot(mu,UY)
        return mu,sigma
        
        
    def _check_convergence(self,mu,mu_old):
        '''
        Checks convergence of Mean Field Approximation
        '''
        if np.sum( abs(mu - mu_old) > self.tol ) == 0:
            return True
        return False
