import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model.base import LinearModel
from sklearn.utils import check_X_y
from scipy.linalg import svd



class VariationalLinearRegression(RegressorMixin,LinearModel):
    '''
    Implements fully Bayesian Linear Regression using mean-field approximation 
    over latent variables. Assumes gamma prior on precision of coefficients 
    and noise.

    Parameters:
    -----------
    n_iter: int, optional (DEFAULT = 10)
       Maximum number of iterations for KL minimization

    tol: float, optional (DEFAULT = 1e-3)
       Convergence threshold
       
    fit_intercept: bool, optional (DEFAULT = True)
       If True will use bias term in model fitting

    a: float, optional (Default = 1e-6)
       Shape parameter of Gamma prior for precision of coefficients
       
    b: float, optional (Default = 1e-6)
       Rate parameter of Gamma prior for precision coefficients
       
    c: float, optional (Default = 1e-6)
       Shape parameter of  Gamma prior for precision of noise
       
    d: float, optional (Default = 1e-6)
       Rate parameter of  Gamma prior for precision of noise
       
    verbose: bool, optional (Default = False)
       If True at each iteration progress report is printed out
       
       
    Attributes
    ----------
    
    
    '''
    
    def __init__(self, n_iter = 300, tol =1e-3, fit_intercept = True, 
                 a = 1e-6, b = 1e-6, c = 1e-6, d = 1e-6, copy_X = True,
                 verbose = False):
        self.n_iter     =  n_iter
        self.tol        =  tol
        self.a,self.b   =  a, b
        self.c,self.d   =  c, d
        self.copy_X     =  copy_X
        self.fit_intercept = fit_intercept
        self.verbose       = verbose

        
        
    def fit(self,X,y):
        '''
        Fits Variational Bayesian Linear Regression Model
        '''
        # preprocess data
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        n_samples, n_features = X.shape
        X, y, X_mean, y_mean, X_std = self._center_data(X, y, self.fit_intercept,
                                                        self.copy_X)
        self._x_mean_  = X_mean
        self._y_mean_  = y_mean
        self._x_std_   = X_std
        self.scores_   = [np.NINF]
        
        # SVD decomposition, done once , reused at each iteration
        u,D,vt = svd(X, full_matrices = False)
        Dsq    = D**2
        
        # compute X'*Y  &  Y'*Y to reuse in each iteration
        XY             = np.dot(X.T,y)
        YY             = np.dot(y,y)
        
        # some parameters of Gamma distribution have closed form solution
        a              = self.a + 0.5 * n_features
        c              = self.c + 0.5 * n_samples
        b,d            = self.b,  self.d
        
        # initial mean of posterior for coefficients
        Mw = 0
                
        for i in range(self.n_iter):
            
            #  ----------   UPDATE Q(w)   --------------
            
            # calculate expected values of alpha and lambda
            e_tau         = self._gamma_mean(c,d)
            e_alpha       = self._gamma_mean(a,b)
            
            # update parameters of Q(w)
            Mw_old       = np.copy(Mw)
            Mw,Sigma     = self._posterior_dist_beta(e_tau, e_alpha,XY,Dsq,vt)
            
            #  ----------    UPDATE Q(alpha_)   ------------
            
            # update rate parameter for Gamma distributed precision of weights
            E_w_sq        = np.dot(Mw,Mw) + np.trace(Sigma)
            b             = self.b + 0.5*E_w_sq
            
            #  ----------    UPDATE Q(lambda_)   ------------

            # update rate parameter for Gamma distributed precision of likelihood 
            # precalculate some values for reuse in lower bound calculation           
            XMw           = np.sum(np.dot(X,Mw)**2)
            XSX           = np.sum(np.dot(X,Sigma)*X)
            MwXY          = np.dot(Mw,XY)
            d             = self.d + 0.5*(YY + XSX + XMw) - MwXY
            
            
            # --------- Convergence Check ---------
            
            if self.verbose is True:
                print "Iteration {0} is completed".format(i)
                
            # check convergence 
            converged = self._check_convergence(Mw,Mw_old)
            
            # save fitted parameters
            if converged or i==(self.n_iter-1):
                break
            
        e_tau   = self._gamma_mean(c,d)
        e_alpha = self._gamma_mean(a,b)
        self.coef_, self.sigma_ = self._posterior_dist_beta(e_tau, e_alpha,XY,
                                                            Dsq,vt)
        self._set_intercept(X_mean,y_mean,X_std)
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
        y_hat  = self._decision_function(X)
        x      = (X - self._x_mean_) / self._x_std_
        
        # asymptotic noise
        noise     = 1./ self._gamma_mean(self._c_,self._d_)
        var       = noise + np.sum(np.dot(x,self.sigma_)*x,axis = 1)
        return [y_hat,var]
        
        
    def _posterior_dist_beta(self, e_tau, e_alpha, XY, Dsq, vt):
        '''
        Calculates parameters of approximation of posterior distribution 
        of weights
        '''
        # inverse eigenvalues of precision matrix
        Dinv = 1. / (e_tau*Dsq + e_alpha)
        # Covariance matrix ( use numpy broadcasting to speed up)
        Sn   = np.dot(vt.T*(Dinv),vt)
        # mean of approximation for posterior distribution
        Mn   = e_tau * np.dot(Sn,XY)
        return [Mn,Sn]
        
        
    def _check_convergence(self,Mw,Mw_old):
        '''
        Checks convergence of Mean Field Approximation
        '''
        if np.sum( abs(Mw - Mw_old) > self.tol ) == 0:
            return True
        return False
        
        
    @staticmethod
    def _gamma_mean(a,b):
       '''
       Calculates mean of gamma distribution
       '''
       return float(a) / b

if __name__ == "__main__":
    x           = np.zeros([100,1])
    x[0:100,0]  = np.linspace(-3,3,100)
    model       = lambda x: 2 + 2*x[:,0]
    y           = model(x) + np.random.normal(0,1,100)
    br2         = VariationalLinearRegression()
    br2.fit(x,y)
    x_test      = np.zeros([20,1])
    x_test[:,0] = np.linspace(-200,200,20)
    y_hat, var  = br2.predict_dist(x_test)
    y_model     = model(x_test)
    
    
    def simple_plotter(x,y, br = None, title = None):
        '''
        Plots mean and variance of predictive distribution
        '''
        # predictive distribution
        y_hat, var = br.predict_dist(x)
        y_lo       = y_hat - 1.96*np.sqrt(var)
        y_hi       = y_hat + 1.96*np.sqrt(var)
        # plotting
        plt.figure(figsize = (12,8))
        plt.plot(x[:,0],y,"b+",label = 'data')
        plt.plot(x[:,0],y_hat,'ro', label = 'model prediction')
        plt.plot(x[:,0],y_lo,'g-', label = 'means +- 1.96*std')
        plt.plot(x[:,0],y_hi,'g-', label = 'means +- 1.96*std')    
        plt.legend(loc = 2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(title)
        plt.show()
        return y_hat, var
        
        
    y1,v1 = simple_plotter(x,y,br2,'predictive distribution, Bayesian Linear Regression')
    y2,v2 = simple_plotter(x_test,y_model,br2,'predictive distribution, Extrapolation')

    
