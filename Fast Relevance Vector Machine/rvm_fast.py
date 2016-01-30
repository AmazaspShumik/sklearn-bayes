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
from sklearn.preprocessing import scale



def update_precisions(Q,S,q,s,A,active,tol):
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
    return [A, converged]


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
        
    compute_score : boolean, optional (DEFAULT = False)
        If True, computes logarithm of marginal likelihood at each iteration.
        (Should be non-decreasing)

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
        
    scores_ : list
        if computed, value of log marginal likelihood 
        
        
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
    
    def __init__( self, n_iter = 500, tol = 1e-3, perfect_fit_tol = 1e-4, 
                  fit_intercept = True, normalize = False, 
                  copy_X = True, verbose = False):
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
        #TODO: how construct predict dist        
        self._X_mean_ = X_mean

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
        A[start]      = XXd[start]/( XY[start] / XXd[start] - var_y)

        for i in range(self.n_iter):
            XXa     = XX[active,:][:,active]
            XYa     = XY[active]
            Aa      =  A[active]
            
            # mean & covariance of posterior distribution
            Mn, Sn  = self._posterior_dist(Aa,beta,XXa,XYa)
            
            # compute quality & sparsity parameters
            bxsn  = beta**2 * np.dot(XX[:,active],Sn)
            bxy   = beta*XY
            bxx   = beta*XXd
            
            s,q,S,Q = self._sparsity_quality(bxsn,bxy,bxx,XYa,XX,X,Sn,
                                                     A,Aa,active,n_samples,
                                                     n_features)
                
            # update precision parameter for noise distribution
            rss     = np.sum( ( y - np.dot(X[:,active] , Mn) )**2 )
            beta    = ( n_samples - np.sum(active) + np.sum(Aa * np.diag(Sn)) )
            beta   /= rss

            # update precision parameters of coefficients
            A,converged  = update_precisions(Q,S,q,s,A,active,self.tol)

            # if converged OR if near perfect fit , them terminate
            if rss / n_samples < self.perfect_fit_tol:
                break
            if converged or i == self.n_iter - 1:
                break
                 
        # after last update of alpha & beta update parameters
        # of posterior distribution
        XXa,XYa,Aa  = XX[active,:][:,active],XY[active],A[active]
        Mn, Sn      = self._posterior_dist(Aa,beta,XXa,XYa)
        
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
        X: numpy array of size [n_samples, n_features]
           Test data, matrix of explanatory variables
           
        Returns
        -------
        y_hat: array of size [n_samples]
           Mean of predictive distribution
           
        std_hat: array of size [n_samples]
           Standard deviation of predictive distribution
        '''
        # mean of predictive distribution
        y_hat     = self.predict(X)
        # variance of predictivee distribution
        x         = (X - self._X_mean_)[:,self.active_]
        var_hat   = self.alpha_
        var_hat  += np.sum( np.dot(x,self.sigma_) * x, axis = 1)
        std_hat   = np.sqrt(var_hat)
        return y_hat, std_hat


    def _posterior_dist(self,A,beta,XX,XY):
        '''
        Calculates mean and covariance matrix of 
        posterior distribution of coefficients
        '''
        # precision matrix 
        Sinv = beta * XX
        np.fill_diagonal(Sinv, np.diag(Sinv) + A)
        # use inversion for symmetric Pos.Def matrices
        Sn   =  pinvh(Sinv)
        Mn   =  beta * np.dot(Sn,XY)
        return [Mn,Sn]
    
    
    def _sparsity_quality(self,bxsn,bxy,bxx,XYa,XX,X,Sn,A,Aa,active,n,m):
        '''
        Calculates sparsity and quality parameters for each feature
        
        Theoretical Note:
        -----------------
        Here we used Woodbury Identity for inverting covariance matrix
        of target distribution 
        C    = 1/beta + 1/alpha * X' * X
        C^-1 = beta - beta^2 * X' * Sn * X
        '''
        Q     = bxy - np.dot( bxsn , XYa )
        # heuristic threshold for much faster calculations
        if n > 2*m:
            S = bxx - np.diag(np.dot(bxsn,XX[active,:]))
        else:
            S = bxx - np.sum( np.dot(bxsn,X[:,active].T) * X.T, axis = 1).T
        qi         = Q
        si         = S # copy not to change S itself
        Qa,Sa      = Q[active], S[active]
        qi[active] = Aa * Qa / (Aa - Sa )
        si[active] = Aa * Sa / (Aa - Sa )
        return [si,qi,S,Q]
        
        
        
#----------------------- Classification ARD -----------------------------------
     
     
def cost_grad(X,Y,w,diagA):
    '''
    Calculates cost and gradient for logistic regression
    
    X: numpy matrix of size 'n x m'
       Matrix of explanatory variables       
       
    Y: numpy vector of size 'n x 1'
       Vector of dependent variables
       
    w: numpy array of size 'm x 1'
       Vector of parameters
       
    diagA: numpy array of size 'm x 1'
       Diagonal of matrix 
    '''
    n     = X.shape[0]
    Xw    = np.dot(X,w)
    s     = expit(Xw)
    si    = 1- s
    wdA   = w*diagA
    cost  = np.sum( -1*np.log(s) * Y - np.log(si)*(1 - Y)) + np.sum(w*wdA)/2
    grad  = np.dot(X.T, s - Y) + wdA
    return [cost/n,grad/n]    
        
        
        
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
            
    compute_score : boolean, optional (DEFAULT = False)
        If True, computes logarithm of marginal likelihood at each iteration.
        (Should be non-decreasing)

    fit_intercept : boolean, optional (DEFAULT = True)
        If True will use intercept in the model. If set
        to false, no intercept will be used in calculations
        
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
        
    lambda_ : float
       estimated precisions of weights
       
    active_ : array, dtype = np.bool, shape = (n_features)
       True for non-zero coefficients, False otherwise

    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients
        
    scores_ : list
        if computed, value of log marginal likelihood 
     
    #TODO
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
    def __init__(self, n_iter = 150, tol = 1e-4, solver = 'lbfgs_b', 
                 n_iter_solver = 30, tol_solver = 1e-5,
                 compute_score = False, fit_intercept = True, normalize = False, 
                 copy_X = True, verbose = False):
        self.n_iter        = n_iter
        self.tol           = tol
        self.solver        = solver
        self.n_iter_solver = n_iter_solver
        self.tol_solver    = tol_solver
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.normalize     = normalize
        self.copy_X        = copy_X
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
        

    def _sparsity_quality(self,X,Xa,y,B,A,Aa,active,Sn):
        XB    = X.T*B
        YB    = y*B
        XSX   = np.dot(np.dot(Xa,Sn),Xa.T)
        bxy   = np.dot(XB,y)        
        Q     = bxy - np.dot( np.dot(XB,XSX), YB)
        S     = np.sum( XB*X.T,1 ) - np.sum( np.dot( XB,XSX )*XB,1 )
        qi    = Q
        si    = S 
        Qa,Sa      = Q[active], S[active]
        qi[active] = Aa * Qa / (Aa - Sa )
        si[active] = Aa * Sa / (Aa - Sa )
        return [si,qi,S,Q]
        
    
    
    def _posterior_dist(self,X,y,A):
        '''
        Uses Laplace approximation for calculating posterior distribution
        '''
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

#        Wmap         = fmin_l_bfgs_b(f, x0 = w_init, pgtol   = self.pgtol_irls,
#                                                   maxiter = self.max_iter_irls)[0]
#        # calculate negative of Hessian at w = Wmap (for Laplace approximation)
#        s            = sigmoid(np.dot(X,Wmap))
#        Z            = s * (1 - s)
#        S            = np.dot(X.T*Z,X)
#        np.fill_diagonal(S,np.diag(S) + diagA)
#        R            = np.linalg.cholesky(S)        
        
        
        return [Mn,Sn,1./B,t_hat]
        
        
    def _preprocess_predictive_x(self,x):
        '''
        Preprocesses test set data matrix before using it in prediction
        '''
        if self.normalize:
            x = (x - self._X_mean) / self._X_std
        if self.fit_intercept:
            x = np.concatenate((np.ones([x.shape[0],1]),x),1)
        return x
            
    
    def predict_proba(self,X):
        x = self._preprocess_predictive_x(X)
        return expit(np.dot(x,self.coef_))
        
    
        







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
    
    compute_score : boolean, optional (DEFAULT = False)
        If True, computes logarithm of marginal likelihood at each iteration.
        (Should be non-decreasing)

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
        
    kernel: str, optional (DEFAULT = 'rbf')
        Type of kernel to be used
    
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
        
    scores_ : list
        if computed, value of log marginal likelihood
        
    relevant_vectors_ : array 
        Relevant Vectors
    
    '''
    def __init__(self, n_iter=1200, tol = 1e-3, perfect_fit_tol = 1e-6, 
                 compute_score = False, fit_intercept = True, normalize = False, 
                 copy_X = True, verbose = False, kernel = 'rbf', degree = 3,
                 gamma  = None, coef0  = 0.1, kernel_params = None):
                     
        super(RVR,self).__init__(n_iter, tol, perfect_fit_tol, compute_score, 
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
        Predict targets
        '''
        K = get_kernel( X, self.relevant_vectors_, self.gamma, self.degree, 
                       self.coef0, self.kernel, self.kernel_params)
        return decision_function(self,self.coef_[self.active_], K,
                                 self.intercept_)
        
        
    def predict_dist(self,X):
        '''
        
        Predictive distribution is calculated for each data point
        
        Returns:
        --------
        '''
        # mean of predictive distribution
        K = get_kernel( X, self.relevant_vectors_, self.gamma, self.degree, 
                       self.coef0, self.kernel, self.kernel_params)
        return super(RVR,self).predict_dist(K)
    
    
    
class RVC(ClassificationARD):
    
    def __init__(self):
        pass
    
    def fit(self):
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

#    n_features = 180
#    n_samples  = 200
#    X      = np.random.random([n_samples,n_features]) + 100
#    X[:,5] = np.linspace(0,10,n_samples)
#    Y      = 20*X[:,5] + 5 + np.random.normal(0,1,n_samples)
#    X,x,Y,y = train_test_split(X,Y,test_size = 0.4)
#    
#    # RegressionARD
#    ard = RegressionARD(n_iter = 20, compute_score = True, verbose = True)
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
    
#    
#    from scipy import stats
#    ###############################################################################
#    # Generating simulated data with Gaussian weights
#    
#
#    # Parameters of the example
#    n_samples, n_features = 800, 800
#    # Create Gaussian data
#    X = np.random.randn(n_samples, n_features)
#    # Create weigts with a precision lambda_ of 4.
#    lambda_ = 1.
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
#    # RegressionARD
#    ard = RegressionARD(fit_intercept = True)
#    start_ard = time.time()
#    ard.fit(X,Y)
#    end_ard   = time.time()
#    y_hat = ard.predict(x)
#    ard_time = end_ard - start_ard
#    
#    print "FAST BAYESIAN LEARNER"
#    print np.sum( ( y - y_hat )**2 ) / n_samples
#    print "VARIATIONAL ARD"
#    print np.sum( ( y - ysk_hat )**2 ) / n_samples
#    print 'timing sklearn {0}, features {1}'.format(sk_time,np.sum(skard.coef_!=0))
#    print 'timing ard sbl {0}, features {1}'.format(ard_time,np.sum(ard.coef_!=0))
    
#    from scipy import stats
#    # Parameters of the example
#    n_samples, n_features = 600, 100
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

#    x          = np.zeros([500,2])
#    x[:,0]     = np.random.normal(0,1,500)
#    x[:,1]     = np.random.normal(0,1,500)
#    x[0:200,0] = x[0:200,0] + 6
#    x[0:200,1] = x[0:200,1] + 2
#    y          = np.ones(500)
#    y[0:200]   = 0
    
#    clf = ClassificationARD(normalize = True)
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
#    lr = LogisticRegression(C = 100)
#    t1 = time.time()
#
#    lr.fit(X,Y)
#    t2 = time.time()
#    y_lr = lr.predict(x)
#    print 'error log reg'
#    print float(np.sum(y_lr!=y)) / y.shape[0]
#    print 'time lr {0}'.format(t2-t1)
    
    
    
    
    
    
    
    
    
    