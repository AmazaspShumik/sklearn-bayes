"""
Models implemented 

"""



import numpy as np
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_X_y,check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model.base import LinearModel, LinearClassifierMixin
from sklearn.linear_model.coordinate_descent import ElasticNet
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import RegressorMixin
    
  
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
                           
                            
                              
class KernelisedElasticNetRegression(LinearModel,RegressorMixin):
    """Linear regression with kernelised features with combined L1 and L2 priors
    as regularizer. 
    
    Using kernel matrix instead of raw features allows to fit more complex models.
    At the same time 


    Parameters
    ----------
    alpha : float
        Constant that multiplies the penalty terms. Defaults to 1.0
        See the notes for the exact mathematical meaning of this
        parameter.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` with the Lasso object is not advised
        and you should prefer the LinearRegression object.
        
    l1_ratio : float
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``. For
        ``l1_ratio = 0`` the penalty is an L2 penalty. ``For l1_ratio = 1`` it
        is an L1 penalty.  For ``0 < l1_ratio < 1``, the penalty is a
        combination of L1 and L2.
        
    fit_intercept : bool
        Whether the intercept should be estimated or not. If ``False``, the
        data is assumed to be already centered.
        
    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.
        
    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument. For sparse input
        this option is always ``True`` to preserve sparsity.
        WARNING : The ``'auto'`` option is deprecated and will
        be removed in 0.18.
        
    max_iter : int, optional
        The maximum number of iterations
        
    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.
        
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
        
    warm_start : bool, optional
        When set to ``True``, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        
    positive : bool, optional
        When set to ``True``, forces the coefficients to be positive.
        
    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.
        
    random_state : int, RandomState instance, or None (default)
        The seed of the pseudo random number generator that selects
        a random feature to update. Useful only when selection is set to
        'random'.
    """
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True,
                 normalize=False, precompute=False, max_iter=1000,
                 copy_X=True, tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic',kernel = 'rbf',
                 degree = 2, gamma = 1, coef0 = 1, kernel_params = None):
        self.alpha         = alpha
        self.l1_ratio      = l1_ratio
        self.coef_         = None
        self.fit_intercept = fit_intercept
        self.normalize     = normalize
        self.precompute    = precompute
        self.max_iter      = max_iter
        self.copy_X        = copy_X
        self.tol           = tol
        self.warm_start    = warm_start
        self.positive      = positive
        self.intercept_    = 0.0
        self.random_state  = random_state
        self.selection     = selection
        self.kernel         = kernel
        self.gamma          = gamma
        self.degree         = degree
        self.coef0          = coef0
        self.kernel_params  = kernel_params
        
        
    def fit(self,X,y):
        '''
        Fits ElasticNet Regression with kernelised features
        
        Parameters
        ----------
        X: array-like of size [n_samples, n_features]
           Matrix of explanatory variables
           
        y: array-like of size (n_samples,)
           Vector of dependent variable
        
        Returns
        -------
        obj: self
          self
        '''
        X,y = check_X_y(X,y, dtype = np.float64)
        K   = get_kernel(X, X, self.gamma, self.degree, self.coef0, self.kernel, 
                         self.kernel_params )
        model = ElasticNet(self.alpha, self.l1_ratio, self.fit_intercept,
                           self.normalize, self.precompute, self.max_iter,
                           self.copy_X, self.tol, self.warm_start, self.positive,
                           self.random_state, self.selection)
        self._model = model.fit(K,y)
        self.relevant_indices_ = np.where(self._model.coef_ != 0)[0]
        #if self.relevant_indices_.shape[0]
        self.relevant_vectors_ = X[self.relevant_indices_,:]
        
        
    def _decision_function(self, X):
        '''
        Decision function for Linear Model
        '''
        check_is_fitted(self, "_model")
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])
        K = get_kernel(X,self.relevant_vectors_,self.gamma, self.degree, 
                       self.coef0, self.kernel, self.kernel_params)
        return safe_sparse_dot(K, self._model.coef_.T[self._model.coef_ != 0],
                               dense_output=True) + self._model.intercept_       
        
           
            
        
class KernelisedLassoRegression(KernelisedElasticNetRegression):
    """Linear Model trained with L1 prior as regularizer (aka the Lasso)

    Parameters
    ----------
    alpha : float, optional
        Constant that multiplies the L1 term. Defaults to 1.0.
        ``alpha = 0`` is equivalent to an ordinary least square, solved
        by the :class:`LinearRegression` object. For numerical
        reasons, using ``alpha = 0`` is with the Lasso object is not advised
        and you should prefer the LinearRegression object.
        
    fit_intercept : boolean
        whether to calculate the intercept for this model. If set
        to false, no intercept will be used in calculations
        (e.g. data is expected to be already centered).
        
    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.
        
    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.
        
    precompute : True | False | 'auto' | array-like
        Whether to use a precomputed Gram matrix to speed up
        calculations. If set to ``'auto'`` let us decide. The Gram
        matrix can also be passed as argument. For sparse input
        this option is always ``True`` to preserve sparsity.
        WARNING : The ``'auto'`` option is deprecated and will
        be removed in 0.18.
    max_iter : int, optional
        The maximum number of iterations
        
    tol : float, optional
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        dual gap for optimality and continues until it is smaller
        than ``tol``.
        
    warm_start : bool, optional
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
        
    positive : bool, optional
        When set to ``True``, forces the coefficients to be positive.
        
    selection : str, default 'cyclic'
        If set to 'random', a random coefficient is updated every iteration
        rather than looping over features sequentially by default. This
        (setting to 'random') often leads to significantly faster convergence
        especially when tol is higher than 1e-4.
        
    random_state : int, RandomState instance, or None (default)
        The seed of the pseudo random number generator that selects
        a random feature to update. Useful only when selection is set to
        'random'.
    """
    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False,
                 precompute=False, copy_X=True, max_iter=1000,
                 tol=1e-4, warm_start=False, positive=False,
                 random_state=None, selection='cyclic',kernel = 'rbf',
                 degree = 2, gamma = 1, coef0 = 1, kernel_params = None):
        super(KernelisedLassoRegression, self).__init__(
              alpha=alpha, l1_ratio=1.0, fit_intercept=fit_intercept,
              normalize=normalize, precompute=precompute, copy_X=copy_X,
              max_iter=max_iter, tol=tol, warm_start=warm_start,
              positive=positive, random_state=random_state, selection=selection, 
              kernel = kernel, degree = degree, gamma = gamma, coef0 = coef0,
              kernel_params = kernel_params)
     
#--------------------------------------- L1VM ------------------------------------------------#
                        
                        
class KernelisedLogisticRegression(LinearClassifierMixin):
    '''
    Logistic Regression with kernelised features and with L1 or L2 regularization. 
    Instead of using Logistic Regression on raw features
    
    Parameters
    ----------
    C: float
        Regularisation parameter
        
    dual: bool, optional (DEFAULT = False)
      
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
    '''
    
    def __init__(self, penalty = "L1", C = 1, fit_intercept = True, intercept_scaling = 1,
                 tol = 1e-3, prune_tol = 1e-5, kernel = 'rbf', degree = 2, gamma = 1, coef0 = 1, 
                 kernel_params = None):
        self.penalty        = penalty
        self.C              = C
        self.fit_intercept  = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.prune_tol      = prune_tol
        self.tol            = tol
        self.kernel         = kernel
        self.gamma          = gamma
        self.degree         = degree
        self.coef0          = coef0
        self.kernel_params  = kernel_params
        
        
        
    def fit(self,X,y):
        '''
        Fits L2VM model
        
        Parameters:
        -----------
        X: numpy array of size 'n x m'
           Matrix of explanatory variables
           
        Y: numpy array of size 'n x '
           Vector of dependent variable
        
        Return
        ------
        obj: self
          self
        '''
        X,y = check_X_y(X,y, dtype = np.float64)
        K   = get_kernel(X, X, self.gamma, self.degree, self.coef0, self.kernel, 
                         self.kernel_params )
        self._model = LogisticRegression( penalty = self.penalty, C = self.C, tol = self.tol, 
                                    fit_inercept = self.fit_intercept,dual = self.dual,
                                     intercept_scaling=self.intercept_scaling)
        self._model = self._model.fit(K,y)
        self.support_vecs_ = X[np.abs(self._model.coef_) > self.prune_tol]
        return self
        
        
    def decision_function(self,X):
        '''
        Computes decision function based on separating hyperplane
        '''
        check_is_fitted(self,"_model")
        X = check_array(X,dtype = np.float64)
        
        
        
        
    def predict_proba(self,X):
        pass

 
               


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    X = np.zeros([100,1])
    X[:,0] = np.linspace(-2,2,100)
    y = np.sinc(X[:,0])
    plt.plot(X[:,0],y)
    klr =KernelisedLassoRegression(kernel = 'rbf', degree =2, alpha = 0.005)
    klr.fit(X,y)    
    y_hat = klr.predict(X)
    plt.plot(X[:,0],y_hat,'b+')
    plt.plot(X[:,0],y_hat,'ro')
    plt.show()
        
        
     