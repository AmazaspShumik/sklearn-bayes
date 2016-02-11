import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.utils.optimize import newton_cg
from scipy.special import expit
from scipy.linalg import pinvh
from scipy.linalg import eigvalsh
from sklearn.utils.multiclass import check_classification_targets
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y
#from sklearn.preprocessing import LabelBinariser
from sklearn.linear_model.logistic import ( _logistic_loss_and_grad, _logistic_loss, 
                                            _logistic_grad_hess,)



class BayesianLogisticRegression(LinearClassifierMixin,BaseEstimator):
    '''
    Implements Bayesian Logistic Regression with type II maximum likelihood, uses
    Gaussian (Laplace) method for approximation of evidence function.
    

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
        
    n_iter_solver: int, optional (DEFAULT = 10)
        Maximum number of iterations before termination of solver
        
    tol_solver: float, optional (DEFAULT = 1e-3)
        Convergence threshold for solver (it is used in estimating posterior
        distribution), 

    fit_intercept : bool, optional ( DEFAULT = True )
        If True will use intercept in the model. If set
        to false, no intercept will be used in calculations
        
    alpha: float (DEFAULT = 1e-6)
        Initial regularization parameter (precision of prior distribution)
        
    verbose : boolean, optional (DEFAULT = True)
        Verbose mode when fitting the model
        
        
    Attributes
    ----------
    coef_ : array, shape = (n_features)
        Coefficients of the regression model (mean of posterior distribution)

    sigma_ : array, shape = (n_features, n_features)
        estimated covariance matrix of the weights, computed only
        for non-zero coefficients
    
    intercept_: array, shape = (n_features)
        intercepts

    
    References:
    -----------
    1) Pattern Recognition and Machine Learning, Bishop (2006) (pages 293 - 294)
    2) Storcheus Dmitry, Mehryar Mohri, Afshin Rostamizadeh. "Foundations of Coupled Nonlinear Dimensionality Reduction."
       arXiv preprint arXiv:1509.08880 (2015).
    '''
    
    def __init__(self, n_iter = 30, tol = 1e-3,solver = 'lbfgs_b',n_iter_solver = 10,
                 tol_solver = 1e-3, fit_intercept = True, alpha = 1e-5,  verbose = False):
        self.n_iter            = n_iter
        self.tol               = tol
        self.n_iter_solver     = n_iter_solver
        self.tol_solver        = tol_solver
        self.verbose           = verbose
        self.fit_intercept     = fit_intercept
        self.alpha             = alpha
        if solver not in ['lbfgs_b','newton_cg']:
            raise ValueError(('Only "lbfgs_b" and "newton_cg" '
                              'solvers are implemented'))
        self.solver            = solver
        
        
    def fit(self,X,y):
        '''
        Fits Bayesian Logistic Regression with Laplace approximation

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
        # preprocess data
        X,y = check_X_y( X, y , dtype = np.float64)
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # take into account bias term if required 
        n_samples, n_features = X.shape
        n_features = n_features + int(self.fit_intercept)
        
        if n_classes < 2:
            raise ValueError("Need samples of at least 2 classes")
        if n_classes > 2:
            self.coef_, self.sigma_ = [0]*n_classes,[0]*n_classes
            self.intercept_         = [0]*n_classes
        else:
            self.coef_, self.sigma_, self.intercept_ = [0],[0],[0]

        for i in range(len(self.coef_)):
            w0 = np.zeros(n_features)
            if n_classes == 2:
                pos_class = self.classes_[1]
            else:
                pos_class = self.classes_[i]
            mask = (y == pos_class)
            y_bin = np.ones(y.shape, dtype=np.float64)
            y_bin[~mask] = -1.
            coef, sigma_ = self._fit(X,y_bin,w0, self.alpha)
            if self.fit_intercept:
                self.intercept_[i] = coef[-1]
                coef_              = coef[:-1]
            self.coef_[i]  = coef_
            self.sigma_[i] = sigma_
            
        self.coef_  = np.asarray(self.coef_)
        return self
        
        
        
    def predict_proba(self,X):
        '''
        Predicts probabilities of targets for test set
        
        Parameters
        ----------
        X: array-like of size [n_samples_test,n_features]
           Matrix of explanatory variables (test set)
           
        Returns
        -------
        probs: numpy array of size [n_samples_test]
           Estimated probabilities of target classes
        '''
        scores = self.decision_function(X)
        sigma = np.asarray([np.sum(np.dot(X,s)*X,axis = 1) for s in self.sigma_])
        ks = 1. / ( 1. + np.pi*sigma / 8)**0.5
        probs = expit(scores.T*ks).T
        if probs.shape[1] == 1:
            probs =  np.hstack([1 - probs, probs])
        else:
            probs /= np.reshape(np.sum(probs, axis = 1), (probs.shape[0],1))
        return probs
        
        
    def _fit(self,X,y,w0, alpha0):
        '''
        Maximizes evidence function (type II maximum likelihood) 
        '''
        # iterative evidence maximization
        alpha = alpha0
        for i in range(self.n_iter):
                        
            # find mean & covariance of Laplace approximation to posterior
            w, d   = self._posterior(X, y, alpha, w0) 
            mu_sq  = np.dot(w,w)
            
            # use EM  to update parameters            
            alpha = X.shape[1] / (mu_sq + np.sum(d)) 
            
            # check convergence
            delta_alpha = abs(alpha - alpha0)
            if delta_alpha < self.tol or i==self.n_iter-1:
                break
            alpha0 = alpha
            
        # after convergence we need to find updated MAP vector of parameters
        # and covariance matrix of Laplace approximation
        coef_, sigma_ = self._posterior(X, y, alpha , w, True)
        return coef_, sigma_
            
            
            
    def _posterior(self, X, Y, alpha0, w0, full_covar = False):
        '''
        Iteratively refitted least squares method using l_bfgs_b.
        Finds MAP estimates for weights and Hessian at convergence point
        '''
        if self.solver == 'lbfgs_b':
            f = lambda w: _logistic_loss_and_grad(w,X,Y,alpha0)
            w = fmin_l_bfgs_b(f, x0 = w0, pgtol = self.tol_solver,
                              maxiter = self.n_iter_solver)[0]
        elif self.solver == 'newton_cg':
            f    = _logistic_loss
            grad = lambda w,*args: _logistic_loss_and_grad(w,*args)[1]
            hess = _logistic_grad_hess               
            args = (X,Y,alpha0)
            w    = newton_cg(hess, f, grad, w0, args=args,
                             maxiter=self.n_iter, tol=self.tol)[0]
        else:
            raise NotImplementedError('Liblinear solver is not yet implemented')
            
        # calculate negative of Hessian at w
        if self.fit_intercept:
            XW = np.dot(X,w[:-1]) + w[-1]
        else:
            XW = np.dot(X,w)
        s          = expit(XW)
        R          = s * (1 - s)
        negHessian = np.dot(X.T*R,X)
        
        # do not regularise constant
        alpha_vec     = np.zeros(negHessian.shape[0])
        alpha_vec     = alpha0   
        np.fill_diagonal(negHessian,np.diag(negHessian) + alpha_vec)
        if full_covar is False:
            eigs = 1./eigvalsh(negHessian)
            return [w,eigs]
        else:
            inv = pinvh(negHessian)
            return [w, inv]
            
            
if __name__ == '__main__':
    import matplotlib.pyplot as plt     
#    # create data set 
#    x          = np.zeros([500,2])
#    x[:,0]     = np.random.normal(0,1,500) 
#    x[:,1]     = np.random.normal(0,1,500) 
#    x[0:250,0] = x[0:250,0] + 3
#    x[0:250,1] = x[0:250,1] + 5
#    #x          = x - np.mean(x,0)
#    #x          = scale(x)
#    y          = -1*np.ones(500)
#    y[0:250]   = 1
#    blr        = BayesianLogisticRegression(solver = 'newton_cg')
#    blr.fit(x,y)
#    
#    # create grid for heatmap
#    n_grid = 500
#    max_x      = np.max(x,axis = 0)
#    min_x      = np.min(x,axis = 0)
#    X1         = np.linspace(min_x[0],max_x[0],n_grid)
#    X2         = np.linspace(min_x[1],max_x[1],n_grid)
#    x1,x2      = np.meshgrid(X1,X2)
#    Xgrid      = np.zeros([n_grid**2,2])
#    Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
#    Xgrid[:,1] = np.reshape(x2,(n_grid**2,))
#    
#    blr_grid   = blr.predict_proba(Xgrid)[:,1]
#    plt.figure(figsize=(8,6))
#    plt.contourf(X1,X2,np.reshape(blr_grid,(n_grid,n_grid)),cmap="coolwarm")
#    plt.plot(x[y==-1,0],x[y==-1,1],"bo", markersize = 3)
#    plt.plot(x[y==1,0],x[y==1,1],"ro", markersize = 3)
#    plt.colorbar()
#    plt.title("Bayesian Logistic Regression, fitted with EM")
#    plt.xlabel("x")
#    plt.ylabel("y")
#    plt.show()

    
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import scale
    from sklearn.cross_validation import train_test_split
    from sklearn.linear_model import LogisticRegressionCV 
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    
    
    # Parameters of the example
    n_samples, n_features = 1000, 1200
    # Create Gaussian data
    np.random.seed(0)
    X = np.random.randn(n_samples, n_features)
    # Create weigts
    lambda_ = 100
    w = np.zeros(n_features)
    # Only 2 relevant features (so that we can vizualise)
    relevant_features = np.random.randint(0, n_features, 2)
    for i in relevant_features:
       w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
    # Create the target
    y = np.dot(X, w) + 10
    y_hat  = np.ones(y.shape[0])
    y_hat[y < 10] = -1
    X,x,Y,y = train_test_split(X,y_hat, test_size = 0.2)
    
    # logistic regression
    lrl2 = LogisticRegressionCV(Cs=[0.01,0.1,1,10,100], penalty = 'l2')
    lrl1 = LogisticRegressionCV(Cs=[0.01,0.1,1,10,100], penalty = 'l1',
                                solver = 'liblinear')
    clf_ard =  BayesianLogisticRegression()
    
    lrl2.fit(X,Y)
    lrl1.fit(X,Y)
    clf_ard.fit(X,Y)
    
    n_grid = 100
    max_x      = np.max(x[:,relevant_features],axis = 0)
    min_x      = np.min(x[:,relevant_features],axis = 0)
    X1         = np.linspace(min_x[0],max_x[0],n_grid)
    X2         = np.linspace(min_x[1],max_x[1],n_grid)
    x1,x2      = np.meshgrid(X1,X2)
    Xgrid      = np.zeros([n_grid**2,2])
    Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
    Xgrid[:,1] = np.reshape(x2,(n_grid**2,))
    Xg         = np.random.randn(n_grid**2,n_features)
    Xg[:,relevant_features[0]] = Xgrid[:,0]
    Xg[:,relevant_features[1]] = Xgrid[:,1]
    
    blr_grid  = clf_ard.predict_proba(Xg)[:,1]
    lrl2_grid = lrl2.predict_proba(Xg)[:,1]
    lrl1_grid = lrl1.predict_proba(Xg)[:,1]
    a1,a2     = relevant_features
    titles = ["ARD Logistic regression","Logistic Regression L2 penalty",
              "Logistic Regression L1 penalty"]
    models  = [blr_grid,lrl2_grid,lrl1_grid]
    
    print "Logistic Regression L1 {0} misclassification rate".format(float(np.sum(y!=lrl1.predict(x))) / x.shape[0])
    print "Logistic Regression L2 {0} misclassification rate".format(float(np.sum(y!=lrl2.predict(x))) / x.shape[0])
    print "ARD Classification {0} misclassification rate".format(float(np.sum(y!=clf_ard.predict(x))) / x.shape[0])
    
    for title,model in zip(titles,models):
       plt.figure(figsize=(8,6))
       plt.contourf(X1,X2,np.reshape(model,(n_grid,n_grid)),cmap="coolwarm")
       plt.plot(x[y==-1,a1],x[y==-1,a2],"bo", markersize = 3)
       plt.plot(x[y==1,a1],x[y==1,a2],"ro", markersize = 3)
       plt.colorbar()
       plt.title(title)
       plt.xlabel("x")
       plt.ylabel("y")
       plt.show()
       
       
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(BayesianLogisticRegression)