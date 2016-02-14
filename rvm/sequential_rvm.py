# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.linear_model.base import LinearModel
from sklearn.utils import check_X_y
from sklearn.utils.extmath import pinvh
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted
from scipy.special import expit
from scipy.optimize import fmin_l_bfgs_b
from sklearn.linear_model import ARDRegression
from scipy.linalg import solve_triangular
from sklearn.utils.optimize import newton_cg
import warnings


def update_precisions(Q,S,q,s,A,active,tol,heuristics = 1e+4):
    '''
    Selects one feature to be added/recomputed/deleted to model based on 
    effect it will have on value of log marginal likelihood.
    '''
    # initialise vector holding changes in log marginal likelihood
    deltaL = np.zeros(Q.shape[0])
    
    # identify features that can be added , recomputed and deleted in model
    theta        =  q**2 - s 
    add          =  (theta > 0) * (active == False)
    recompute    =  (theta > 0) * (active == True)
    delete       = ~(add + recompute)
    
    # compute sparsity & quality parameters corresponding to features in 
    # three groups identified above
    Qadd,Sadd      = Q[add], S[add]
    Qrec,Srec,Arec = Q[recompute], S[recompute], A[recompute]
    Qdel,Sdel,Adel = Q[delete], S[delete], A[delete]
    
    # compute new alpha's (precision parameters) for features that are 
    # currently in model and will be recomputed
    Anew           = s[recompute]**2/theta[recompute]
    delta_alpha    = (1./Anew - 1./Arec)
    
    # compute change in log marginal likelihood 
    deltaL[add]       = ( Qadd**2 - Sadd ) / Sadd + np.log(Sadd/Qadd**2 )
    deltaL[recompute] = Qrec**2 / (Srec + 1. / delta_alpha) - np.log( 1 + Srec*delta_alpha)
    deltaL[delete]    = Qdel**2 / (Sdel - Adel) - np.log(1 - Sdel / Adel)
    
    # find feature which caused largest change in likelihood
    feature_index = np.argmax(deltaL)
             
    # no deletions or additions
    same_features  = np.sum( theta[~recompute] > 0) == 0
    
    # changes in precision for features already in model is below threshold
    no_delta       = np.sum( abs( Anew - Arec ) > tol ) == 0
    
    # check convergence: if features to add or delete and small change in 
    #                    precision for current features then terminate
    converged      = False
    if same_features and no_delta:
        converged = True
        return [A,converged]
    
    # if not converged update precision parameter of weights and return
    if theta[feature_index] > 0:
        A[feature_index] = s[feature_index]**2 / theta[feature_index]
        if active[feature_index] == False:
            active[feature_index] = True
    else:   
        if active[feature_index] == True and np.sum(active) > 1:
            active[feature_index] = False
            A[feature_index]      = np.PINF
    return [A,converged]


###############################################################################
#                ARD REGRESSION AND CLASSIFICATION
###############################################################################


#-------------------------- Regression ARD ------------------------------------


class RegressionARD(LinearModel,RegressorMixin):
    '''
    Regression with Automatic Relevance Determination. 
    
    
    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 100)
        Maximum number of iterations
        
    tol: float, optional (DEFAULT = 1e-3)
        If absolute change in precision parameter for weights is below threshold
        algorithm terminates.
    
    perfect_fit_tol: float, optional (DEFAULT = 1e-4)
        Algortihm terminates in case MSE on training set is below perfect_fit_tol.
        Helps to prevent overflow of precision parameter for noise in case of
        nearly perfect fit.

    fit_intercept : boolean, optional (DEFAULT = True)
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
        
    normalize : boolean, optional (DEFAULT = False)
        If True, the regressors X will be normalized before regression
        
    copy_X : boolean, optional (DEFAULT = True)
        If True, X will be copied; else, it may be overwritten.
        
    verbose : boolean, optional (DEFAULT = True)
        Verbose mode when fitting the model
        
        
    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)
        
    alpha_ : float
       estimated precision of the noise
       
    active_ : array, dtype = np.bool, shape = (n_features)
       True for non-zero coefficients, False otherwise
       
    lambda_ : array, shape = (n_features)
       estimated precisions of the coefficients
       
    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients      
      
      
    Methods
    -------
    fit:           fits model
    predict:       calculates prediction for targets on test set
    predict_dist:  calculates parameters of predictive distribution on test set
    
      
    Examples
    --------
    >>> clf = RegressionARD()
    >>> clf.fit([[0,0], [1, 1], [2, 2]], [0, 1, 2])
    ... # doctest: +NORMALIZE_WHITESPACE
    RegressionARD(compute_score=False, copy_X=True, fit_intercept=True, n_iter=100,
       normalize=False, perfect_fit_tol=0.001, verbose=False)
    >>> clf.predict([[1, 1]])
    array([ 1.])
        
    '''
    
    def __init__( self, n_iter = 300, tol = 1e-1, perfect_fit_tol = 1e-4, 
                  fit_intercept = True, normalize = False, copy_X = True,
                  verbose = False):
        self.n_iter          = n_iter
        self.tol             = tol
        self.perfect_fit_tol = perfect_fit_tol
        self.scores_         = list()
        self.fit_intercept   = fit_intercept
        self.normalize       = normalize
        self.copy_X          = copy_X
        self.verbose         = verbose
    
        
    def fit(self,X,y):
        '''
        Fits ARD Regression with Sequential Sparse Bayes Algorithm.
        This is 
        
        Parameters
        -----------
        X: numpy array of size [n_samples, n_features]
           Training data, matrix of explanatory variables
        
        y: numpy array of size [n_samples, n_features] 
           Target values
        '''
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        n_samples, n_features = X.shape
        
        X, y, X_mean, y_mean, X_std = self._center_data(X, y, self.fit_intercept,
                                                        self.normalize, self.copy_X)
        self._x_mean_ = X_mean
        self._y_mean  = y_mean
        self._x_std   = X_std

        #  precompute X'*Y , X'*X for faster iterations & allocate memory for
        #  sparsity & quality vectors
        XY     = np.dot(X.T,y)
        XX     = np.dot(X.T,X)
        XXd    = np.diag(XX)

        #  initialise precision of noise & and coefficients
        var_y  = np.var(y)
        beta   = 1. / np.var(y)
        A      = np.PINF * np.ones(n_features)
        active = np.zeros(n_features , dtype = np.bool)
        
        # start from a single basis vector with largest projection on targets
        proj  = XY**2 / XXd
        start = np.argmax(proj)
        active[start] = True
        A[start]      = XXd[start]/( proj[start] - var_y)

        for i in range(self.n_iter):
            
            XXa     = XX[active,:][:,active]
            XYa     = XY[active]
            Aa      =  A[active]
            
            # mean & covariance of posterior distribution
            Mn,Ri  = self._posterior_dist(Aa,beta,XXa,XYa)
            Sdiag  = np.sum(Ri**2,0)
            
            # compute quality & sparsity parameters            
            s,q,S,Q = self._sparsity_quality(XX,XXd,XY,XYa,Aa,Ri,active,beta)
                
            # update precision parameter for noise distribution
            rss     = np.sum( ( y - np.dot(X[:,active] , Mn) )**2 )
            beta    = n_samples - np.sum(active) + np.sum(Aa * Sdiag )
            beta   /= rss

            # update precision parameters of coefficients
            A,converged  = update_precisions(Q,S,q,s,A,active,self.tol)
            
            if self.verbose:
                print(('Iteration: {0}, number of features '
                       'in the model: {1}').format(i,np.sum(active)))
            

            # if converged OR if near perfect fit , them terminate
            if rss / n_samples < self.perfect_fit_tol:
                warnings.warn('Early termination due to near perfect fit')
                break
            
            if converged or i == self.n_iter - 1:
                if converged and self.verbose:
                    print('Algorithm converged !')
                break
                 
        # after last update of alpha & beta update parameters
        # of posterior distribution
        XXa,XYa,Aa  = XX[active,:][:,active],XY[active],A[active]
        Mn, Sn      = self._posterior_dist(Aa,beta,XXa,XYa,True)
        self.coef_         = np.zeros(n_features)
        self.coef_[active] = Mn
        self.sigma_        = Sn
        self.active_       = active
        self.lambda_       = A
        self.alpha_        = beta
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
        X: numpy array of size [n_samples_test, n_features]
           Test data, matrix of explanatory variables
           
        Returns
        -------
        y_hat: array of size [n_samples_test]
           Estimated values of targets on test set (Mean of predictive distribution)
           
        std_hat: array of size [n_samples_test]
           Error bounds (Standard deviation of predictive distribution)
        '''
        x         = (X - self._x_mean_) / self._x_std
        y_hat     = np.dot(x,self.coef_) + self._y_mean 
        var_hat   = self.alpha_
        var_hat  += np.sum( np.dot(x,self.sigma_) * x, axis = 1)
        std_hat   = np.sqrt(var_hat)
        return y_hat, std_hat


    def _posterior_dist(self,A,beta,XX,XY,full_covar = False):
        '''
        Calculates mean and covariance matrix of 
        posterior distribution of coefficients
        '''
        # precision matrix 
        Sinv = beta * XX
        np.fill_diagonal(Sinv, np.diag(Sinv) + A)
        R    = np.linalg.cholesky(Sinv)
        Z    = solve_triangular(R,beta*XY, check_finite = False, lower = True)
        Mn   = solve_triangular(R.T,Z,check_finite = False, lower = False)
        Ri   = solve_triangular(R,np.eye(A.shape[0]), check_finite = False, lower = True)
        if full_covar:
            Sn   = np.dot(Ri.T,Ri)
            return Mn,Sn
        else:
            return Mn,Ri
    
    
    def _sparsity_quality(self,XX,XXd,XY,XYa,Aa,Ri,active,beta):
        '''
        Calculates sparsity and quality parameters for each feature
        
        Theoretical Note:
        -----------------
        Here we used Woodbury Identity for inverting covariance matrix
        of target distribution 
        C    = 1/beta + 1/alpha * X' * X
        C^-1 = beta - beta^2 * X' * Sn * X
        '''
        bxy        = beta*XY
        bxx        = beta*XXd
        xr         = np.dot(XX[:,active],Ri.T)
        S          = bxx - beta**2 * np.sum( xr**2, axis=1)
        Q          = bxy - beta**2 * np.dot( xr, np.dot(Ri,XYa))
        qi         = np.copy(Q)
        si         = np.copy(S) 
        Qa,Sa      = Q[active], S[active]
        qi[active] = Aa * Qa / (Aa - Sa )
        si[active] = Aa * Sa / (Aa - Sa )
        return [si,qi,S,Q]
              
        
#----------------------- Classification ARD -----------------------------------
     
     
def _logistic_cost_grad(X,Y,w,diagA):
    '''
    Calculates cost and gradient for logistic regression
    '''
    n     = X.shape[0]
    Xw    = np.dot(X,w)
    s     = expit(Xw)
    si    = 1- s
    wdA   = w*diagA
    cost  = np.sum( -1*np.log(s) * Y - np.log(si)*(1 - Y)) + np.sum(w*wdA)/2
    grad  = np.dot(X.T, s - Y) + wdA
    return [cost/n,grad/n]
    

def _logistic_cost_grad_hess(X,Y,w,diagA):
    '''
    Calculates cost, gradient and hessian for logistic regression
    '''
    pass


def _logistic_cost():
    '''
    Calculates cost of logistic regression
    '''
    pass
        
        
        
class ClassificationARD(LinearModel,ClassifierMixin):
    '''
    Logistic Regression with Automatic Relevance determination
    
    
    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 100)
        Maximum number of iterations before termination
        
    tol: float, optional (DEFAULT = 1e-3)
        If absolute change in precision parameter for weights is below threshold
        algorithm terminates.
        
    solver: str, optional (DEFAULT = 'lbfgs_b')
        Optimization method that is used for finding parameters of posterior
        distribution ['lbfgs_b','newton_cg']
        
    n_iter_solver: int, optional (DEFAULT = 20)
        Maximum number of iterations before termination of solver
        
    tol_solver: float, optional (DEFAULT = 1e-5)
        Convergence threshold for solver (it is used in estimating posterior
        distribution), 

    fit_intercept : boolean, optional (DEFAULT = True)
        If True will use intercept in the model. If set
        to false, no intercept will be used in calculations
        
    normalize : boolean, optional (DEFAULT = False)
        If True, the regressors X will be normalized before regression
        

    verbose : boolean, optional (DEFAULT = True)
        Verbose mode when fitting the model
        
        
    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)
        
    lambda_ : float
       estimated precisions of weights
       
    active_ : array, dtype = np.bool, shape = (n_features)
       True for non-zero coefficients, False otherwise

    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients

    '''
    def __init__(self, n_iter = 300, tol = 1e-4, solver = 'lbfgs_b', 
                 n_iter_solver = 30, tol_solver = 1e-5, fit_intercept = True,
                 normalize = False, verbose = False):
        self.n_iter        = n_iter
        self.tol           = tol
        self.solver        = solver
        self.n_iter_solver = n_iter_solver
        self.tol_solver    = tol_solver
        self.fit_intercept = fit_intercept
        self.normalize     = normalize
        self.verbose       = verbose
    
    
    def fit(self,X,y):
        '''
        Fits Logistic Regression with ARD
        
        Parameters
        ----------
        X: numpy array of size [n_samples, n_features]
           Training data, matrix of explanatory variables
        
        y: numpy array of size [n_samples] 
           Target values
        '''
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        n_samples, n_features = X.shape
        
        # normalise X and add vector of intercepts
        if self.normalize:
            self._X_mean, self._X_std = np.mean(X,0), np.std(X,0)
            X = (X - self._X_mean) / self._X_std
        
        if self.fit_intercept:
            X = np.concatenate((np.ones([n_samples,1]),X),1)
            n_features += 1
        
        A         = np.PINF * np.ones(n_features)
        active    = np.zeros(n_features , dtype = np.bool)
        active[0] = True
        A[0]      = 1e-6
        
        for i in range(self.n_iter):
            Xa      = X[:,active]
            Aa      =  A[active]
            
            # mean & covariance of posterior distribution
            Mn,Sn,B,t_hat = self._posterior_dist(Xa,y, Aa)
            
            # compute quality & sparsity parameters
            s,q,S,Q = self._sparsity_quality(X,Xa,t_hat,B,A,Aa,active,Sn)

            # update precision parameters of coefficients
            A,converged  = update_precisions(Q,S,q,s,A,active,self.tol)

            # terminate if converged
            if converged or i == self.n_iter - 1:
                break
        
        Xa,Aa   = X[:,active], A[active]
        Mn,Sn,B,t_hat = self._posterior_dist(Xa,y,Aa)
        self.coef_   = np.zeros(n_features)
        self.active_ = active
        self.coef_[self.active_]   = Mn
        self.sigma_  = Sn
        self.lambda_ = A
        
        
                
    
    def predict_proba(self,X):
        '''
        Predicts probabilities of targets for test set
        
        Parameters
        ----------
        
        
        '''
        x = self._preprocess_predictive_x(X)
        return expit(np.dot(x,self.coef_))
        

    def _sparsity_quality(self,X,Xa,y,B,A,Aa,active,Sn):
        XB    = X.T*B
        YB    = y*B
        XSX   = np.dot(np.dot(Xa,Sn),Xa.T)
        bxy   = np.dot(XB,y)        
        Q     = bxy - np.dot( np.dot(XB,XSX), YB)
        S     = np.sum( XB*X.T,1 ) - np.sum( np.dot( XB,XSX )*XB,1 )
        qi    = np.copy(Q)
        si    = np.copy(S) 
        Qa,Sa      = Q[active], S[active]
        qi[active] = Aa * Qa / (Aa - Sa )
        si[active] = Aa * Sa / (Aa - Sa )
        return [si,qi,S,Q]
        
    
    def _posterior_dist(self,X,y,A):
        '''
        Uses Laplace approximation for calculating posterior distribution
        '''
        if self.solver == 'lbfgs_b':
            f  = lambda w: cost_grad(X,y,w,A)
            w_init  = np.random.random(X.shape[1])
            Mn      = fmin_l_bfgs_b(f, x0 = w_init, pgtol = self.tol_solver,
                                    maxiter = self.n_iter_solver)[0]
            Xm      = np.dot(X,Mn)
            s       = expit(Xm)
            B       = s * (1 - s)
            S       = np.dot(X.T*B,X) 
            np.fill_diagonal(S, np.diag(S) + A)
            t_hat   = Xm + (y - s)*1./B
            Sn      = pinvh(S)
        elif self.solver == 'newton_cg':
            

#        Wmap         = fmin_l_bfgs_b(f, x0 = w_init, pgtol   = self.pgtol_irls,
#                                                   maxiter = self.max_iter_irls)[0]
#        # calculate negative of Hessian at w = Wmap (for Laplace approximation)
#        s            = sigmoid(np.dot(X,Wmap))
#        Z            = s * (1 - s)
#        S            = np.dot(X.T*Z,X)
#        np.fill_diagonal(S,np.diag(S) + diagA)
#        R            = np.linalg.cholesky(S)        
        return [Mn,Sn,B,t_hat]
        
        
    def _preprocess_predictive_x(self,x):
        '''
        Preprocesses test set data matrix before using it in prediction
        '''
        if self.normalize:
            x = (x - self._X_mean) / self._X_std
        if self.fit_intercept:
            x = np.concatenate((np.ones([x.shape[0],1]),x),1)
        return x

        

###############################################################################
#                  Relevance Vector Machine: RVR and RVC
###############################################################################



def get_kernel( X, Y, gamma, degree, coef0, kernel, kernel_params ):
    '''
    Calculates kernelised features for RVR and RVC
    '''
    if callable(kernel):
        params = kernel_params or {}
    else:
        params = {"gamma": gamma,
                  "degree": degree,
                  "coef0": coef0  }
    return pairwise_kernels(X, Y, metric=kernel,
                            filter_params=True, **params)
                            
           
                 
def decision_function(estimator , active_coef_ , K , intercept_ ):
    '''
    Computes decision function for regression and classification.
    In regression case output is mean of predictive distribution
    In classification case decision function calculates value of 
    separating hyperplane for data points in K
       
    '''
    check_is_fitted(estimator, "coef_")
    return np.dot(K,active_coef_) + intercept_
    
    

class RVR(RegressionARD):
    '''
    Relevance Vector Regression is ARD regression with kernelised features
    
    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 100)
        Maximum number of iterations

    fit_intercept : boolean, optional (DEFAULT = True)
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
        
    scale : boolean, optional (DEFAULT = False)
        If True, the regressors X will be normalized before regression
        
    copy_X : boolean, optional (DEFAULT = True)
        If True, X will be copied; else, it may be overwritten.
        
    verbose : boolean, optional (DEFAULT = True)
        Verbose mode when fitting the model 
        
    kernel: str, optional (DEFAULT = 'rbf')
        Type of kernel to be used (all kernels: ['rbf' | 'poly' | 'sigmoid']
    
    degree : int, (DEFAULT = 3)
        Degree for poly kernels. Ignored by other kernels.
        
    gamma : float, optional (DEFAULT = 1/n_features)
        Kernel coefficient for rbf and poly kernels, ignored by other kernels
        
    coef0 : float, optional (DEFAULT = 0.1)
        Independent term in poly and sigmoid kernels, ignored by other kernels
        
    kernel_params : mapping of string to any, optional
        Parameters (keyword arguments) and values for kernel passed as
        callable object, ignored by other kernels
        
        
    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)
        
    alpha_ : float
       estimated precision of the noise
       
    active_ : array, dtype = np.bool, shape = (n_features)
       True for non-zero coefficients, False otherwise
       
    lambda_ : array, shape = (n_features)
       estimated precisions of the coefficients
       
    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients
        
    relevant_vectors_ : array 
        Relevant Vectors
    
    '''
    def __init__(self, n_iter=300, tol = 1e-3, perfect_fit_tol = 1e-6, 
                 fit_intercept = True, scale = False, copy_X = True,
                 verbose = False, kernel = 'rbf', degree = 3, gamma  = None,
                 coef0  = 1, kernel_params = None):
        # !!! do not normalise kernel matrix
        normalize = False
        super(RVR,self).__init__(n_iter, tol, perfect_fit_tol, 
                                 fit_intercept, normalize, copy_X, verbose)
        self.kernel = kernel
        self.degree = degree
        self.gamma  = gamma
        self.coef0  = coef0
        self.kernel_params = kernel_params
    
    
    def fit(self,X,y):
        '''
        Fit Relevance Vector Regression Model
        
        Parameters
        -----------
        X: numpy array of size [n_samples, n_features]
           Training data, matrix of explanatory variables
        
        y: numpy array of size [n_samples, n_features] 
           Target values
        '''
        # if gamma is not defined set it to default values
        if self.gamma == None:
            self.gamma = 1. / X.shape[0]
            
        # kernelise features
        K = get_kernel( X, X, self.gamma, self.degree, self.coef0, 
                       self.kernel, self.kernel_params)
                       
        # use fit method of RegressionARD
        _ = super(RVR,self).fit(K,y)
        self.relevant_vectors_ = X[self.active_,:]
        return self
        
        
    def predict(self,X):
        '''
        Predicts targets
        
        Parameters
        ----------
        X: numpy array of size [n_samples_test, n_features]
           Training data, matrix of explanatory variables
           
        Returns
        --------
         : numpy array of size [n_samples_test]
           Estimated target values on test set 
        '''
        K = get_kernel( X, self.relevant_vectors_, self.gamma, self.degree, 
                       self.coef0, self.kernel, self.kernel_params)
        return decision_function(self,self.coef_[self.active_], K,
                                 self.intercept_)
        
        
    def predict_dist(self,X):
        '''
        Computes predictive distribution for test set.
        Predictive distribution for each data point is one dimensional
        Gaussian and therefore is characterised by mean and standard
        deviation.
        
        Parameters
        ----------
        X: numpy array of size [n_samples_test, n_features]
           Training data, matrix of explanatory variables
           
        Returns
        -------
        y_hat: array of size [n_samples_test]
           Estimated values of targets on test set (Mean of predictive distribution)
           
        std_hat: array of size [n_samples_test]
           Error bounds (Standard deviation of predictive distribution)
        '''
        # mean of predictive distribution
        K = get_kernel( X, self.relevant_vectors_, self.gamma, self.degree, 
                       self.coef0, self.kernel, self.kernel_params)
        y_hat     = decision_function(self,self.coef_[self.active_], K, self.intercept_)
        var_hat   = self.alpha_
        var_hat  += np.sum( np.dot(K,self.sigma_) * K, axis = 1)
        std_hat   = np.sqrt(var_hat)
        return y_hat,std_hat
    
    
    
class RVC(ClassificationARD):
    
    def __init__(self, n_iter = 300, tol = 1e-4, solver = 'lbfgs_b', 
                 n_iter_solver = 30, tol_solver = 1e-5,
                 compute_score = False, fit_intercept = True, normalize = False, 
                 copy_X = True, verbose = False, kernel = 'rbf', degree = 3,
                 gamma  = None, coef0  = 1, kernel_params = None):
                     
        super(RVC,self).__init__(n_iter = 300, tol = 1e-4, solver = 'lbfgs_b', 
             n_iter_solver = 30, tol_solver = 1e-5,
             compute_score = False, fit_intercept = True, normalize = False, 
             copy_X = True, verbose = False)
        self.kernel = kernel
        self.degree = degree
        self.gamma  = gamma
        self.coef0  = coef0
        self.kernel_params = kernel_params
        
        
    def fit(self,X):
        '''
        Fits relevance vector classification
        '''
        pass
    
    
    def predict(self,X):
        K = get_kernel( X, self.relevant_vectors_, self.gamma, self.degree, 
                       self.coef0, self.kernel, self.kernel_params)
        hyperplane = decision_function(self,self.coef_[self.active_], K,
                                       self.intercept_)
        # predict targets using values of separating hyperplane
    
    
    def predict_proba(self,X):
        K = get_kernel( X, self.relevant_vectors_, self.gamma, self.degree, 
                       self.coef0, self.kernel, self.kernel_params)
        return super(RVR,self).predict_proba(K)
        
    
    
if __name__ == "__main__":
    from sklearn.cross_validation import train_test_split
    import time
#
#    n_features = 180
#    n_samples  = 400
#    X      = np.random.random([n_samples,n_features])
#    X[:,0] = np.linspace(0,10,n_samples)
#    Y      = 20*X[:,0] + 5 + np.random.normal(0,1,n_samples)
#    X,x,Y,y = train_test_split(X,Y,test_size = 0.4)
#    
#    # RegressionARD
#    ard = RegressionARD(n_iter = 200, verbose = True)
#    start_ard = time.time()
#    ard.fit(X,Y)
#    end_ard   = time.time()
#    y_hat = ard.predict(x)
#    ard_time = end_ard - start_ard
#    
#    # sklearn ARD
#    skard = ARDRegression()
#    start_skard = time.time()
#    skard.fit(X,Y)
#    end_skard   = time.time()
#    ysk_hat = skard.predict(x)
#    sk_time = end_skard - start_skard
#    
#    print "FAST BAYESIAN LEARNER"
#    print np.sum( (y - y_hat)**2 ) / n_samples
#    print "VARIATIONAL ARD"
#    print np.sum( (y - ysk_hat)**2 ) / n_samples
#    
#    import matplotlib.pyplot as plt
#    plt.plot(x[:,1],y_hat,'ro')
#    plt.plot(x[:,1],y,'b+')
#    plt.show()
#    
#    plt.plot(y_hat - y,'go')
#    plt.plot(ysk_hat - y,'ro')
#    plt.show()
#    
#    print 'timing sklearn {0}'.format(sk_time)
#    print 'timing ard sbl {0}'.format(ard_time)
#    
#    from sklearn.utils.testing import assert_array_almost_equal
#    def test_toy_ard_object():
#        # Test BayesianRegression ARD classifier
#        X = np.array([[1], [2], [3]])
#        Y = np.array([1, 2, 3])
#        clf = RegressionARD(compute_score=True)
#        clf.fit(X, Y)
#    
#        # Check that the model could approximately learn the identity function
#        test = [[1], [3], [4]]
#        assert_array_almost_equal(clf.predict(test), [1, 3, 4], 2)
#        
#    test_toy_ard_object()
    
    
#    from scipy import stats
#    ###############################################################################
#    # Generating simulated data with Gaussian weights
#    import rvm_fast
#
#    # Parameters of the example
#    n_samples, n_features = 1000, 800
#    # Create Gaussian data
#    X = np.random.randn(n_samples, n_features)
#    # Create weigts with a precision lambda_ of 4.
#    lambda_ = 4.
#    w = np.zeros(n_features)
#    # Only keep 10 weights of interest
#    relevant_features = np.random.randint(0, n_features, 10)
#    for i in relevant_features:
#        w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
#    # Create noite with a precision alpha of 50.
#    alpha_ = 1.
#    noise = stats.norm.rvs(loc=0, scale=1 / np.sqrt(alpha_) , size=n_samples)
#    # Create the target
#    y = np.dot(X, w) + noise
#    X,x,Y,y = train_test_split(X,y, test_size = 0.2)
#    
#    # sklearn ARD
#    skard = ARDRegression()
#    start_skard = time.time()
#    skard.fit(X,Y)
#    end_skard   = time.time()
#    ysk_hat = skard.predict(x)
#    sk_time = end_skard - start_skard
#    
#    
#    # RegressionARD blazing fast
#    ard = RegressionARD(fit_intercept = True, n_iter = 300, verbose = True)
#    start_ard = time.time()
#    ard.fit(X,Y)
#    end_ard   = time.time()
#    y_hat,var_hat = ard.predict_dist(x)
#    ard_time = end_ard - start_ard
#    
#    # just fast
#    ard1 = rvm_fast.RegressionARD(fit_intercept = True, n_iter = 300)
#    start_ard = time.time()
#    ard1.fit(X,Y)
#    end_ard   = time.time()
#    y_hat1 = ard1.predict(x)
#    ard1_time = end_ard - start_ard
#    
#    print "BlAZING FAST ARD"
#    print np.sum( ( y - y_hat )**2 ) / n_samples
#    print 'FAST ARD'
#    print np.sum( ( y - y_hat1 )**2 ) / n_samples
#    print "VARIATIONAL ARD"
#    print np.sum( ( y - ysk_hat )**2 ) / n_samples
#    print 'timing ard blazing {0}, features {1}'.format(ard_time,np.sum(ard.coef_!=0))
#    print 'timing ard fast {0}, features {1}'.format(ard1_time,np.sum(ard1.coef_!=0))
#    print 'timing sklearn {0}, features {1}'.format(sk_time,np.sum(skard.coef_!=0))

#    from scipy import stats
#    # Parameters of the example
#    n_samples, n_features = 600, 600
#    # Create Gaussian data
#    X = np.random.randn(n_samples, n_features)
#    # Create weigts with a precision lambda_ of 4.
#    lambda_ = .2
#    w = np.zeros(n_features)
#    # Only keep 10 weights of interest
#    relevant_features = np.random.randint(0, n_features, 10)
#    for i in relevant_features:
#        w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
#    # Create noite with a precision alpha of 50.
#    # Create the target
#    y = np.dot(X, w)
#    y[y > 0] = 1
#    y[y < 0] = 0
#    X,x,Y,y = train_test_split(X,y, test_size = 0.2)
#
#    
#    clf = ClassificationARD(normalize = False)
#    t1 = time.time()
#    clf.fit(X,Y)
#    t2 = time.time()
#    pr = clf.predict_proba(x)
#    y_hat = np.zeros(y.shape[0])
#    y_hat[pr>0.5] = 1
#    print 'ERRor ARD'
#    print float(np.sum(y_hat!=y)) / y.shape[0]
#    print 'time ard {0}'.format(t2-t1)
#    
#    from sklearn.linear_model import LogisticRegression
#    lr = LogisticRegression(C = 1)
#    t1 = time.time()
#
#    lr.fit(X,Y)
#    t2 = time.time()
#    y_lr = lr.predict(x)
#    print 'error log reg'
#    print float(np.sum(y_lr!=y)) / y.shape[0]
#    print 'time lr {0}'.format(t2-t1)
    
    
    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cross_validation import train_test_split
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import mean_squared_error
    
#    # parameters
#    n = 1500
#    
#    # generate data set
#    #np.random.seed(0)
#    Xc       = np.ones([n,1])
#    Xc[:,0]  = np.linspace(-5,5,n)
#    Yc       = 10*np.sinc(Xc[:,0]) + np.random.normal(0,1,n)
#    X,x,Y,y  = train_test_split(Xc,Yc,test_size = 0.5, random_state = 0)
#    
#    # train rvm with fixed-point optimization
#    rvm = RVR(gamma = 1)
#    t1 = time.time()
#    rvm.fit(X,Y)
#    t2 = time.time()
#    y_hat,var     = rvm.predict_dist(x)
#    rvm_err   = mean_squared_error(y_hat,y)
#    rvs       = np.sum(rvm.active_)
#    print "RVM error on test set is {0}, number of relevant vectors is {1}, time {2}".format(rvm_err, rvs, t2 - t1)
#    from sklearn.svm import SVR
#    from sklearn.grid_search import GridSearchCV
#    svr = GridSearchCV(SVR(gamma = 1), param_grid = {'C':[0.001,0.1,1,10,100]}, cv = 5)
#    t1 = time.time()
#    svr.fit(X,Y)
#    t2 = time.time()
#    svm_err = mean_squared_error(svr.predict(x),y)
#    svs     = svr.best_estimator_.support_vectors_.shape[0]
#    print "SVM error on test set is {0}, number of support vectors is {1}, time {2}".format(svm_err, svs, t2 - t1)
#
#    
#    # plot test vs predicted data
#    plt.figure(figsize = (12,8))
#    plt.plot(x[:,0],y,"b+",markersize = 3, label = "test data")
#    plt.plot(x[:,0],y_hat,"rD", markersize = 3, label = "mean of predictive distribution")
#    # plot one standard deviation bounds
#    plt.plot(x[:,0],y_hat + np.sqrt(var),"co", markersize = 3, label = "y_hat +- std")
#    plt.plot(x[:,0],y_hat - np.sqrt(var),"co", markersize = 3)
#    plt.plot(rvm.relevant_vectors_,Y[rvm.active_],"co",markersize = 11,  label = "relevant vectors")
#    plt.legend()
#    plt.title("RVM")
#    plt.show()
    
#    ##########################################################
#    
#    n_iter = 100
#    from sklearn.svm import SVR
#    from sklearn.preprocessing import scale
#    
#    
#    def compare_rvr_svr(X,Y,kernel, gamma, coef0, degree, test_size):
#        '''
#        Compares perfomance of RVR and SVR
#        
#        #TODO: use timeit for timing
#        '''
#        X,x,Y,y = train_test_split(X,Y,test_size = test_size)
#        # RVR
#        rvm = RVR(gamma = gamma, coef0 = coef0, degree = degree, kernel = kernel)
#        t1 = time.time()
#        rvm.fit(X,Y)
#        t2 = time.time()
#        rvr_time = t2 - t1
#        y_hat = rvm.predict(x)
#        rvm_err   = mean_squared_error(y_hat,y)
#        rvs       = np.sum(rvm.active_)
#        # SVR
#        svm = SVR(gamma = gamma, coef0 = coef0, degree = degree, kernel = kernel)
#        svr = GridSearchCV(svm,param_grid = {'C':[0.01,0.1,1,10,100]}, cv = 5)
#        t1 = time.time()
#        svr.fit(X,Y)
#        t2 = time.time()
#        svm_time = t2 - t1
#        svm_err = mean_squared_error(svr.predict(x),y)
#        svs     = svr.best_estimator_.support_vectors_.shape[0]
#        return {'RVR':[rvm_err,rvr_time,rvs],
#                'SVR':[svm_err,svm_time,svs]}
#                
#        
#    from sklearn.datasets import load_boston
#    boston = load_boston()
#    Xb,yb  = scale(boston['data']),boston['target']
#    
#    from sklearn.datasets import load_diabetes
#    diabetes = load_diabetes()
#    Xd,yd  = scale(diabetes['data']),diabetes['target']
#    
#    
#    #b = compare_rvr_svr(Xb,yb,kernel = 'poly', gamma = 1, coef0 = 1, degree=3,
#    #                   test_size = 0.2)
#                        
#    d =  compare_rvr_svr(Xb,yb,kernel = 'rbf', gamma = 0.1, coef0 = 1, degree=2,
#                        test_size = 0.1)
        
    