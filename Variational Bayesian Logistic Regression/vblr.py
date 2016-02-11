# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.utils.optimize import newton_cg
from scipy.special import expit
from scipy.linalg import pinvh, eigvalsh
from scipy.linalg import eigvalsh
from sklearn.utils.multiclass import check_classification_targets
from sklearn.linear_model.base import LinearClassifierMixin, BaseEstimator
from sklearn.utils import check_X_y

#----------------------- Helper functions ----------------------------------


def lam(eps):
    ''' Calculates lambda eps '''
    return 0.5 / eps * ( expit(eps) - 0.5 )
    



class VariationalLogisticRegression(LinearClassifierMixin, BaseEstimator):
    '''
    Variational Bayesian Logistic Regression 
    
    Parameters:
    -----------
    n_iter: int, optional (DEFAULT = 300 )
       Maximum number of iterations
       
    tol: float, optional (DEFAULT = 1e-3)
       Convergence threshold, if cange in coefficients is less than threshold
       algorithm is terminated
    
    fit_intercept: bool, optinal ( DEFAULT = True )
       If True uses bias term in model fitting
       
    a: float, optional (DEFAULT = 1e-6)
       rate
       
    b: float, optional (DEFAULT = 1e-6)
    
    
    verbose: bool, optional (DEFAULT = False)
       Verbose mode

      
    References:
    -----------
    Bishop 2006, Pattern Recognition and Machine Learning ( Chapter 10 )
    Murphy 2012, Machine Learning A Probabilistic Perspective ( Chapter 21 )
    '''
    def __init__(self,  n_iter = 300, tol = 1e-3, fit_intercept = True,
                 a = 1e-6, b = 1e-6, verbose = True):
        self.n_iter            = n_iter
        self.tol               = tol
        self.verbose           = verbose
        self.fit_intercept     = fit_intercept
        self.a                 =  a
        self.b                 =  b
        
        
    def fit(self,X,y):
        '''
        Fits variational Bayesian Logistic Regression
        
        Parameters
        ----------
        X: array-like of size [n_samples, n_features]
           Matrix of explanatory variables
           
        y: array-like of size [n_samples]
           Vector of dependent variables

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
        if self.fit_intercept:
            X = np.hstack( (np.ones([n_samples,1]),X))
        
        # handle multiclass problems using One-vs-Rest 
        if n_classes < 2:
            raise ValueError("Need samples of at least 2 classes")
        if n_classes > 2:
            self.coef_, self.sigma_ = [0]*n_classes,[0]*n_classes
            self.intercept_         = [0]*n_classes
        else:
            self.coef_, self.sigma_, self.intercept_ = [0],[0],[0]
        
        # huperparameters of 
        a  = self.a + 0.5*n_features
        b  = self.b
        
        for i in range(len(self.coef_)):
            if n_classes == 2:
                pos_class = self.classes_[1]
            else:
                pos_class   = self.classes_[i]
            mask            = (y == pos_class)
            y_bin           = np.ones(y.shape, dtype=np.float64)
            y_bin[~mask]    = 0
            coef_, sigma_  = self._fit(X,y_bin,a,b)
            intercept_ = 0
            if self.fit_intercept:
                intercept_  = coef_[0]
                coef_       = coef_[1:]
            self.coef_[i]   = coef_
            self.intercept_[i] = intercept_
            self.sigma_[i]  = sigma_
        self.coef_  = np.asarray(self.coef_)
        self.sigma_ = np.asarray(self.sigma_)
        return self


            
    def _fit(self,X,y,a,b):
        '''
        Fits single classifier for each class (for OVR framework)
        '''
        eps = 1
        XY  = np.dot( X.T , (y-0.5))
        w0  = np.zeros(X.shape[1])
  
        for i in range(self.n_iter):
            # In the E-stpe we update approximation of 
            # posterior distribution q(w,alpha) = q(w)*q(alpha)
            
            # --------- update q(w) ------------------
            l  = lam(eps)
            print "a, b  =  {0}, {1}".format(a,b)
            w,sigma = self._posterior_dist(X,l,a,b,XY)
            
            print "w^2, trace_sigma , XY = {0}, {1}, {2}".format(np.sum(w**2),np.trace(sigma), np.sum(XY))
            
            # -------- update q(alpha) ---------------
            
            E_w_sq = np.outer(w,w) + sigma
            b = self.b + np.sum(w**2) + np.trace(sigma)#0.5*np.trace(E_w_sq)
            
            # In the M-step we update parameter eps which controls 
            # accuracy of local variational approximation
            eps = np.sqrt( np.sum( np.dot(X,E_w_sq)*X, axis = 1))
            
            # convergence
            if np.sum(abs(w-w0) > self.tol) == 0 or i==self.n_iter-1:
                break
            w0 = w
            
        l  = lam(eps)
        coef_, sigma_  = self._posterior_dist(X,l,a,b,XY)

        return coef_, sigma_


    def _posterior_dist(self,X,l,a,b,XY):
        '''
        Finds gaussian approximation to posterior of coefficients
        '''
        sigma_inv  = 2*np.dot(X.T*l,X)
        alpha_vec  = np.ones(X.shape[1])*float(a) / b
        if self.fit_intercept:
            alpha_vec[0] = 0
        np.fill_diagonal(sigma_inv, np.diag(sigma_inv) + alpha_vec)
        sigma_   = pinvh(sigma_inv)
        mean_    = np.dot(sigma_,XY)     
        return [mean_, sigma_]
        
    
    def predict_proba(self,x):
        pass
        
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt     
#    # create data set 
#    x          = np.zeros([500,2])
#    x[:,0]     = np.random.normal(0,1,500) -3
#    x[:,1]     = np.random.normal(0,1,500) -3
#    x[0:250,0] = x[0:250,0] + 12
#    x[0:250,1] = x[0:250,1] + 1
#    #x          = x - np.mean(x,0)
#    #x          = scale(x)
#    y          = -1*np.ones(500)
#    y[0:250]   = 1
#    blr        = VariationalLogisticRegression(n_iter = 50, fit_intercept = True)
#    blr.fit(x,y)
#    y_hat      = blr.predict(x) 
#    
#    
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
#    blr_grid   = blr.decision_function(Xgrid)
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
    n_samples, n_features = 500, 400
    # Create Gaussian data
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
    X[y_hat>0,:] = X[y_hat>0,:] + 0.2
    X,x,Y,y = train_test_split(X,y_hat, test_size = 0.2)
    
    # logistic regression
    lrl2 = LogisticRegressionCV(Cs=[1e+10], penalty = 'l2')
    lrl1 = LogisticRegressionCV(Cs=[0.01,0.1], penalty = 'l1',
                                solver = 'liblinear')
    clf_ard =  VariationalLogisticRegression(n_iter = 20)
    
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
    
    blr_grid  = clf_ard.decision_function(Xg)
    lrl2_grid = lrl2.decision_function(Xg)
    lrl1_grid = lrl1.decision_function(Xg)
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
