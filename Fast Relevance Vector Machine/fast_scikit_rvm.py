# -*- coding: utf-8 -*-

import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.linear_model.base import LinearModel, LinearClassifierMixin
from sklearn.utils import check_X_y,check_array
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.extmath import pinvh,safe_sparse_dot,log_logistic
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.validation import check_is_fitted, NotFittedError
from scipy.special import expit
from scipy.optimize import fmin_l_bfgs_b
from sklearn.linear_model import ARDRegression
from scipy.linalg import solve_triangular
from sklearn.utils.optimize import newton_cg
import scipy.sparse
import warnings


def update_precisions(Q,S,q,s,A,active,tol, clf_bias = True):
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
    deltaL[recompute] = Qrec**2 / (Srec + 1. / delta_alpha) - np.log(1 + Srec*delta_alpha)
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
        if not clf_bias:
            active_min = 1
        else:
            active_min = 2
        if active[feature_index] == True and np.sum(active) > active_min:
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
        
        Parameters
        -----------
        X: {array-like, sparse matrix} of size [n_samples, n_features]
           Training data, matrix of explanatory variables
        
        y: array-like of size [n_samples, n_features] 
           Target values
           
        Returns
        -------
        self : object
            Returns self.
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
            A,converged  = update_precisions(Q,S,q,s,A,active,self.tol,True)
            
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
        X: {array-like, sparse} [n_samples_test, n_features]
           Test data, matrix of explanatory variables
           
        Returns
        -------
        y_hat: numpy array of size [n_samples_test]
           Estimated values of targets on test set (Mean of predictive distribution)
           
        std_hat: numpy array of size [n_samples_test]
           Error bounds (Standard deviation of predictive distribution)
        '''
        x         = (X - self._x_mean_) / self._x_std
        y_hat     = np.dot(x,self.coef_) + self._y_mean 
        var_hat   = self.alpha_
        var_hat  += np.sum( np.dot(x[:,self.active_],self.sigma_) * x[:,self.active_], axis = 1)
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
     
     
def _logistic_cost_grad(X,Y,w,diagA, penalise_intercept):
    '''
    Calculates cost and gradient for logistic regression
    '''
    n     = X.shape[0]
    Xw    = np.dot(X,w)
    s     = expit(Xw)
    si    = 1 - s
    wdA   = w*diagA
    if not penalise_intercept:
        wdA[0] = 0
    cost  = np.sum( -1*np.log(s)*Y - np.log(si)*(1 - Y)) + np.sum(w*wdA)/2
    grad  = np.dot(X.T, s - Y) + wdA
    return [cost/n,grad/n]
    

# TODO: cost, grad , hessian function for Newton CG
def _logistic_cost_grad_hess(X,Y,w,diagA):
    '''
    Calculates cost, gradient and hessian for logistic regression
    '''
    raise NotImplementedError('to be done')

        
        
        
class ClassificationARD(BaseEstimator,LinearClassifierMixin):
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

    fit_intercept : bool, optional ( DEFAULT = True )
        If True will use intercept in the model. If set
        to false, no intercept will be used in calculations
        
    penalise_intercept: bool, optional ( DEFAULT = False)
        If True uses prior distribution on bias term (penalises intercept)
        
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
    def __init__(self, n_iter = 100, tol = 1e-4, solver = 'lbfgs_b', 
                 n_iter_solver = 15, tol_solver = 1e-4, fit_intercept = True,
                 penalise_intercept = False, normalize = False, verbose = False):
        self.n_iter             = n_iter
        self.tol                = tol
        self.solver             = solver
        self.n_iter_solver      = n_iter_solver
        self.tol_solver         = tol_solver
        self.fit_intercept      = fit_intercept
        self.penalise_intercept = penalise_intercept
        self.normalize          = normalize
        self.verbose            = verbose
    
    
    def fit(self,X,y):
        '''
        Fits Logistic Regression with ARD
        
        Parameters
        ----------
        X: array-like of size [n_samples, n_features]
           Training data, matrix of explanatory variables
        
        y: array-like of size [n_samples] 
           Target values
           
        Returns
        -------
        self : object
            Returns self.
        '''
        X, y = check_X_y(X, y, accept_sparse = None, dtype=np.float64)
        n_samples, n_features = X.shape

        # preprocess features
        self._X_mean = np.zeros(n_features)
        self._X_std  = np.ones(n_features)
        if self.normalize:
            self._X_mean, self._X_std = np.mean(X,0), np.std(X,0)
        X = (X - self._X_mean) / self._X_std
        if self.fit_intercept:
            X = np.concatenate((np.ones([n_samples,1]),X),1)
            n_features += 1
        
        # preprocess targets
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("Need samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % self.classes_[0])
        
        # if multiclass use OVR (i.e. fit classifier for each class)
        self.coef_,self.active_ ,self.lambda_= list(),list(),list()
        self.intercept_, self.sigma_ = list(),list()            
        for pos_class in self.classes_:
            if n_classes == 2:
                pos_class = self.classes_[1]
            mask = (y == pos_class)
            y_bin = np.zeros(y.shape, dtype=np.float64)
            y_bin[mask] = 1
            coef_, intercept_, active_ , sigma_ , A  = self._fit(X,y_bin,
                                                       n_samples,n_features)
            self.coef_.append(coef_)
            self.active_.append(active_)
            self.intercept_.append(intercept_)
            self.sigma_.append(sigma_)
            self.lambda_.append(A)
            # in case of binary classification fit only one classifier           
            if n_classes == 2:
                break
            
        return self
        
    
    def _fit(self,X,y,n_samples,n_features):
        '''
        Fits binary classification
        '''
        A         = np.PINF * np.ones(n_features)
        active    = np.zeros(n_features , dtype = np.bool)
        # this is done for RVC (so that to have at least one rv in worst case)
        if n_features > 1:
            active[1] = True
            A[1]      = 1e-3
        else:
            active[1] = True
            A[1]      = 1e-3
        
        penalise  = self.fit_intercept and self.penalise_intercept
        for i in range(self.n_iter):
            Xa      =  X[:,active]
            Aa      =  A[active]
            penalise_intercept  = active[0] and penalise
            
            # mean & covariance of posterior distribution
            Mn,Sn,B,t_hat = self._posterior_dist(Xa,y, Aa, penalise_intercept)
            
            # compute quality & sparsity parameters
            s,q,S,Q = self._sparsity_quality(X,Xa,t_hat,B,A,Aa,active,Sn)

            # update precision parameters of coefficients
            A,converged  = update_precisions(Q,S,q,s,A,active,self.tol,self.fit_intercept)

            # terminate if converged
            if converged or i == self.n_iter - 1:
                break
        
        penalise_intercept = penalise and active[0]
        Xa,Aa   = X[:,active], A[active]
        Mn,Sn,B,t_hat = self._posterior_dist(Xa,y,Aa,penalise_intercept)
        intercept_ = 0
        if self.fit_intercept:
           n_features -= 1
           if active[0] == True:
               intercept_  = Mn[0]
               Mn          = Mn[1:]               
           active          = active[1:]
        coef_           = np.zeros([1,n_features])
        coef_[0,active] = Mn   
        return coef_, intercept_, active, Sn, A
        
        
    def decision_function(self,X):
        '''
        Decision function
        '''
        check_is_fitted(self, 'coef_') 
        X = check_array(X, accept_sparse=None, dtype = np.float64)
        n_features = self.coef_[0].shape[1]
        if X.shape[1] != n_features:
            raise ValueError("X has %d features per sample; expecting %d"
                             % (X.shape[1], n_features))
        x = (X - self._X_mean) / self._X_std
        decision = np.array([ (np.dot(x,w.T) + c)[:,0] for w,c 
                               in zip(self.coef_,self.intercept_) ]).T
        if decision.shape[1] == 1:
            return decision[:,0]
        return decision
        
        
    def predict(self,X):
        '''
        Calculates estimated target values on test set
        
        Parameters
        ----------
        X: array-like of size [n_samples_test, n_features]
           Matrix of explanatory variables (test set)
           
        Returns
        -------
        y_pred: numpy arra of size [n_samples_test]
           Predicted values of targets
        '''
        probs   = self.predict_proba(X)
        indices = np.argmax(probs, axis = 1)
        y_pred  = self.classes_[indices]
        return y_pred

                 

    def predict_proba(self,X):
        '''
        Predicts probabilities of targets for test set
        Uses probit function to approximate convolution 
        of sigmoid and Gaussian.
        
        Parameters
        ----------
        X: array-like of size [n_samples_test,n_features]
           Matrix of explanatory variables (test set)
           
        Returns
        -------
        probs: numpy array of size [n_samples_test]
           Estimated probabilities of target classes
        '''
        y_hat = self.decision_function(X)
        X     = check_array(X, accept_sparse = None)
        x     = (X - self._X_mean) / self._X_std
        if self.fit_intercept:
            x    = np.concatenate((np.ones([x.shape[0],1]), x),1)
        if y_hat.ndim == 1:
            pr   = self._predict_proba(x[:,self.lambda_[0]!=np.PINF],
                                           y_hat,self.sigma_[0])
            prob = np.vstack([1 - pr, pr]).T
        else:
            pr   = [self._predict_proba(x[:,idx != np.PINF],y_hat[:,i],
                        self.sigma_[i]) for i,idx in enumerate(self.lambda_) ]
            pr   = np.asarray(pr).T
            prob = pr / np.reshape(np.sum(pr, axis = 1), (pr.shape[0],1))
        return prob

        
    def _predict_proba(self,X,y_hat,sigma):
        '''
        Calculates predictive distribution
        '''
        var = np.sum(np.dot(X,sigma)*X,1)
        ks  = 1. / ( 1. + np.pi * var/ 8)**0.5
        pr  = expit(y_hat * ks)
        return pr
        

    def _sparsity_quality(self,X,Xa,y,B,A,Aa,active,Sn):
        '''
        Calculates sparsity & quality parameters for each feature
        '''
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
        
    
    def _posterior_dist(self,X,y,A,intercept_prior):
        '''
        Uses Laplace approximation for calculating posterior distribution
        '''
        if self.solver == 'lbfgs_b':
            f  = lambda w: _logistic_cost_grad(X,y,w,A,intercept_prior)
            w_init  = np.random.random(X.shape[1])
            Mn      = fmin_l_bfgs_b(f, x0 = w_init, pgtol = self.tol_solver,
                                    maxiter = self.n_iter_solver)[0]
            Xm      = np.dot(X,Mn)
            s       = expit(Xm)
            B       = (1-s) * s
            S       = np.dot(X.T*B,X)
            np.fill_diagonal(S, np.diag(S) + A)
            t_hat   = Xm + (y - s)*1./B
            Sn      = pinvh(S)
        elif self.solver == 'newton_cg':
            # TODO: Implement Newton-CG
            raise NotImplementedError(('Newton Conjugate Gradient optimizer '
                                       'is not currently supported'))
        return [Mn,Sn,B,t_hat]
        


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
                            
           
                 
def decision_function(estimator , active_coef_ , X , intercept_,
                      relevant_vectors_, gamma, degree, coef0,
                      kernel,kernel_params):
    '''
    Computes decision function for regression and classification.
    '''
    K = get_kernel( X, relevant_vectors_, gamma, degree, coef0, 
                    kernel, kernel_params)
    return np.dot(K,active_coef_) + intercept_
   



class RVR(RegressionARD):
    '''
    Relevance Vector Regression is ARD regression with kernelised features
    
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
    
    perfect_fit_tol: float, optional (DEFAULT = 1e-4)
        Algortihm terminates in case MSE on training set is below perfect_fit_tol.
        Helps to prevent overflow of precision parameter for noise in case of
        nearly perfect fit.
        
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
    def __init__(self, n_iter=300, tol = 1e-3, perfect_fit_tol = 1e-4, 
                 fit_intercept = True, copy_X = True,verbose = False,
                 kernel = 'poly', degree = 3, gamma  = 1,
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
        X: {array-like,sparse matrix} of size [n_samples, n_features]
           Training data, matrix of explanatory variables
        
        y: array-like of size [n_samples, n_features] 
           Target values
           
        Returns
        -------
        self: object
           self
        '''
        X,y = check_X_y(X,y, accept_sparse = ['csr','coo','bsr'], dtype = np.float64)
        # kernelise features
        K = get_kernel( X, X, self.gamma, self.degree, self.coef0, 
                       self.kernel, self.kernel_params)
        # use fit method of RegressionARD
        _ = super(RVR,self).fit(K,y)
        # convert to csr (need to use __getitem__)
        convert_tocsr = [scipy.sparse.coo.coo_matrix, scipy.sparse.dia.dia_matrix,
                         scipy.sparse.bsr.bsr_matrix]
        if type(X) in convert_tocsr:
            X = X.tocsr()
        self.relevant_  = np.where(self.active_== True)[0]
        if X.ndim == 1:
            self.relevant_vectors_ = X[self.relevant_]
        else:
            self.relevant_vectors_ = X[self.relevant_,:]
        return self
        
        
    def predict(self,X):
        '''
        Predicts targets on test set
        
        Parameters
        ----------
        X: {array-like,sparse_matrix} of size [n_samples_test, n_features]
           Matrix of explanatory variables (test set)
           
        Returns
        --------
         : numpy array of size [n_samples_test]
           Estimated target values on test set 
        '''
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        return self._decision_function(X)
        
        
    def predict_dist(self,X):
        '''
        Computes predictive distribution for test set.
        Predictive distribution for each data point is one dimensional
        Gaussian and therefore is characterised by mean and standard
        deviation.
        
        Parameters
        ----------
        X: {array-like,sparse matrix} of size [n_samples_test, n_features]
           Matrix of explanatory variables (test set)
           
        Returns
        -------
        y_hat: array of size [n_samples_test]
           Estimated values of targets on test set (Mean of predictive distribution)
           
        std_hat: array of size [n_samples_test]
           Error bounds (Standard deviation of predictive distribution)
        '''
        check_is_fitted(self, "coef_")
        # mean of predictive distribution
        K = get_kernel( X, self.relevant_vectors_, self.gamma, self.degree, 
                       self.coef0, self.kernel, self.kernel_params)
        y_hat     = decision_function(self,self.coef_[self.active_], K, self.intercept_)
        var_hat   = self.alpha_
        var_hat  += np.sum( np.dot(K,self.sigma_) * K, axis = 1)
        std_hat   = np.sqrt(var_hat)
        return y_hat,std_hat
              
                                 
    def _decision_function(self,X):
        '''
        Decision function, calculates mean of predicitve distribution
        '''
        check_is_fitted(self, "coef_")
        return decision_function(self , self.coef_[self.active_] ,
                                 X , self.intercept_, self.relevant_vectors_, 
                                 self.gamma, self.degree, self.coef0, self.kernel,
                                 self.kernel_params)
    


class RVC(ClassificationARD):
    '''
    Relevance Vector Classifier
        
    
    Parameters
    ----------
    n_iter: int, optional (DEFAULT = 300)
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

    fit_intercept : bool, optional ( DEFAULT = True )
        If True will use intercept in the model. If set
        to false, no intercept will be used in calculations
        
    penalise_intercept: bool, optional ( DEFAULT = False)
        If True uses prior distribution on bias term (penalises intercept)
        
    normalize : boolean, optional (DEFAULT = False)
        If True, the regressors X will be normalized before regression
        
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
                 verbose = False, kernel = 'rbf', degree = 2,
                 gamma  = None, coef0  = 1, kernel_params = None):
        # use constructor of Classification ARD
        super(RVC,self).__init__(n_iter = 300, tol = 1e-4, solver = 'lbfgs_b', 
                                 n_iter_solver = 30, tol_solver = 1e-5, 
                                 fit_intercept = True, normalize = False,
                                 verbose = False)
        self.kernel = kernel
        self.degree = degree
        self.gamma  = gamma
        self.coef0  = coef0
        self.kernel_params = kernel_params
        self.full_kernel   = False
        
        
    def fit(self,X,y):
        '''
        Fit Relevance Vector Classifier
        
        Parameters
        -----------
        X: array-like of size [n_samples, n_features]
           Training data, matrix of explanatory variables
        
        y: array-like of size [n_samples, n_features] 
           Target values
           
        Returns
        -------
        self: object
           self
        '''
        X,y = check_X_y(X,y, accept_sparse = None, dtype = np.float64)
        # kernelise features
        K = get_kernel( X, X, self.gamma, self.degree, self.coef0, 
                       self.kernel, self.kernel_params)
        # use fit method of RegressionARD
        _ = super(RVC,self).fit(K,y)
        self.relevant_  = [np.where(active==True)[0] for active in self.active_]
        if X.ndim == 1:
            self.relevant_vectors_ = [ X[relevant_] for relevant_ in self.relevant_]
        else:
            self.relevant_vectors_ = [ X[relevant_,:] for relevant_ in self.relevant_ ]
        return self
        
        
    def decision_function(self,X):
        '''
        Decision function
        '''
        check_is_fitted(self, "coef_")
        X = check_array(X, accept_sparse = None, dtype = np.float64) 
        f = lambda coef,x,rv,act,c: decision_function(self,coef[:,act==True].T,x,
                                  c ,rv, self.gamma, self.degree,
                                  self.coef0, self.kernel, self.kernel_params)
        decision = np.asarray([ f(coef,X,rv,act,c)[:,0] for coef,rv,act,c in zip(self.coef_,
                                 self.relevant_vectors_,self.active_,self.intercept_) ]).T
        if decision.shape[1] == 1:
            return decision[:,0]
        return decision
        

    def predict_proba(self,X):
        '''
        Predicts probabilities of targets for test set
        
        Theoretical Note
        ================
        Current version of method does not use MacKay's approximation
        to convolution of Gaussian and sigmoid. This results in less accurate 
        estimation of class probabilities and therefore possible increase
        in misclassification error for multiclass problems (prediction accuracy
        for binary classification problems is not changed)
        
        Parameters
        ----------
        X: array-like of size [n_samples_test,n_features]
           Matrix of explanatory variables (test set)
           
        Returns
        -------
        probs: numpy array of size [n_samples_test]
           Estimated probabilities of target classes
        
        '''
        prob = expit(self.decision_function(X))
        if prob.ndim == 1:
            prob = np.vstack([1 - prob, prob]).T
        prob = prob / np.reshape(np.sum(prob, axis = 1), (prob.shape[0],1))
        return prob
        
        
    def predict(self,X):
        '''
        Predict function
        
        Parameters
        ----------
        X: array-like of size [n_samples_test,n_features]
           Matrix of explanatory variables (test set)
           
        Returns
        -------
        y_pred: numpy array of size [n_samples_test]
           Estimated values of targets
        '''
        probs   = self.predict_proba(X)
        indices = np.argmax(probs, axis = 1)
        y_pred  = self.classes_[indices]
        return y_pred
