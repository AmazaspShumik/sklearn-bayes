# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.linalg import solve_triangular

#----------------------------------  RVM  -----------------------------------------#


class SparseBayesianLearner(object):
    '''
    Implements Sparse Bayesian Learner, in case no kernel is given this is equivalent
    to regression or classwith automatic relevance determination, if kernel is given it is
    equivalent to relevance vector machine (see Tipping 2001).
    
    Parameters:
    -----------
    alpha_max: float
       If alpha corresponding to basis vector will be above alpha_max, then basis
       vector is pruned (i.e. not used in further computations)
       
    thresh: float
       Convergence parameter
       
    kernel: str or None
       Type of kernel to use [currently possible: None, 'gaussian', 'poly']
       
    scaler: float (used for kernels)
       Scaling parameters for kernels
       
    learn_type: str (DEFAULT = "regression")
       Type of learning to be implemented ('regression' or 'classification')
       
    method: str
       Method to fit evidence approximation, currently available: "EM","fixed-point"
       ( Note in 2009 Tipping proposed much faster algorithm for fitting RVM , it is 
       not implemented in this version of code)
       
    max_iter: int
       Maximum number of iterations for evidence maximization
       
    verbose: str
       If True prints messages about progress in computation
       
    max_iter_irls: int
       Maximum number of iterations for IRLS (works in case of regression)
       
       
       
    References:
    -----------
    1) Tipping 2001, Sparse Bayesian Learning and Relevance Vector Machine
    2) Bishop 2006, Pattern Recognition and Machine Learning (Chapters 3,6,7,9)
    3) Storcheus Dmitry, Mehryar Mohri, and Afshin Rostamizadeh. "Foundations of Coupled Nonlinear Dimensionality Reduction."
       arXiv preprint arXiv:1509.08880 (2015).
    '''
    
    def __init__(self, alpha_max = 1e+3, thresh  = 1e-5, kernel        = None,
                                                          scaler        = None,
                                                          learn_type    = 'regression',
                                                          method        = 'fixed-point',
                                                          max_iter      = 100,
                                                          max_iter_irls = 20,
                                                          pgtol_irls    = 1e-3,
                                                          p_order       = 2,
                                                          verbose       = False):
        self.verbose         = verbose
        self.kernel          = kernel
        self.scaler          = scaler 
        # if polynomial kernel, add order
        self.p_order         = p_order
        
        # convergence parameters & maximum allowed number of iterations
        self.conv_thresh     = thresh
        self.alpha_max       = alpha_max
        self.max_iter        = max_iter
        
        # method for evidence approximation , either "EM" or "fixed-point"
        self.method          = method
        
        # type of learning 'classification' or 'regression'
        assert learn_type in ['regression','classification'], ' Undefined parameter "learn_type" '
        self.learn_type      = learn_type
        # parameters for IRLS in case of classification
        self.pgtol_irls      = pgtol_irls
        self.max_iter_irls   = max_iter_irls
        
        # parameters computed while fitting model
        self.Mu              = 0
        self.Sigma           = 0
        self.active          = 0
        self.diagA           = 0
        self.gamma           = 0
           
        
    def fit(self,X,Y):
        '''
        Fits Sparse Bayesian Learning Algorithm, writes mean and covariance of
        posterior distribution to instance variables. 
        
        Parameters:
        -----------        
        
        X: numpy array of size 'n x m'
           Matrix of explanatory variables
       
        Y: numpy vector of size 'n x 1'
           Vector of dependent variables
           
        '''
        self.Xraw            = X
        
        # kernelise data if used for RVM     
        if self.kernel is not None:
           X                 = SparseBayesianLearner.kernel_estimation(X,X,self.kernel, 
                                                                                 self.scaler,
                                                                                 self.p_order)        
        # add bias term
        X                    = np.concatenate((np.ones([X.shape[0],1]),X),1)
        # dimensionality of data
        self.n, self.m       = X.shape
        
        # preprocess dependent variable in case of classification
        if self.learn_type  == "classification":
            classes = set(Y)
            assert len(classes)==2, 'This implementation of RVM can only handle 2 class classification'
            Y       = self._binarise(Y,classes)
            
        # initialise evidence, coeffs & precision parameters for prior & likelihood
        diagA    = np.random.random(self.m)
        beta     = np.random.random()
        
        for i in range(self.max_iter):
                        
            # set of features that will be used in computation
            active      = diagA < self.alpha_max
            self.m      = np.sum(active)
            if self.m == 0:
                raise ValueError("All features were pruned. Check parameter alpha_max")
                        
            # calculate posterior mean & diagonal of covariance matrix ( with EM method 
            # for evidence approximation this corresponds to E-step ). Note that we
            # do not need whole covariance matrix! only diagonal elements.
            Wmap,Sdiag    = self._posterior_params(X[:,active],Y, diagA[active],
                                                                  beta)
            # error term is calculated only for regression case
            if self.learn_type == "regression":
                err       = Y - np.dot(X[:,active],Wmap)
                err_sq    = np.dot(err,err)
            
            # save previous values of alpha, beta 
            old_A         = np.copy(diagA)
            old_beta      = beta
            
            if self.method == "fixed-point":

                # update precision parameters of likelihood & prior
                gamma   = 1 - diagA[active]*Sdiag
                diagA[active] = gamma/Wmap**2
                
                if self.learn_type == "regression":
                   beta          = (self.n - np.sum(gamma))/err_sq
                
            elif self.method == "EM":
                
                # M-step , finds new hyperparameters that maximise likelihood
                diagA[active] = 1.0 / (Wmap**2 + Sdiag)
                if self.learn_type == "regression":
                   gamma   = 1 - diagA[active]*Sdiag
                   beta    = self.n /(err_sq + np.sum(gamma)/beta)
                   
            if self.verbose:
                print "iteration {0} is completed ".format(i)
                
            # if change in alpha & beta is below threshold then terminate iterations
            delta_alpha = np.max(abs(old_A[active] - diagA[active]))
            if delta_alpha < self.conv_thresh:
                if self.learn_type == "classification":
                   break
                else:
                   delta_beta  = abs(old_beta - beta)
                   if delta_beta < self.conv_thresh:
                      break
            if i==self.max_iter:
                print "Warning!!! Algorithm did not converge"
                
            
        self.active          = diagA < self.alpha_max
        self.diagA           = diagA
        self.beta            = beta
        self.m               = np.sum(self.active)
        
        # posterior mean and covariance after last update of alpha & beta
        # after convergence there is only small number of features saved as 
        # 'relevant vectors', so finding full inverse of precision matrix is not
        # costly
        self.Wmap,self.Sigma = self._posterior_params(X[:, self.active],Y,
                                                                        diagA[self.active],
                                                                        beta,
                                                                        True)
      
      
    def predict(self,x):
        '''
        Returns point estimate of targret value (in case of regression returns
        mean of predictive distribution, in case of classification returns estiated 
        target value)
        
        Parameters:
        -----------
        
        x: numpy array of size 'unknown x m'
           Matrix of test explanatory variables.
           
        Returns:
        --------
        
        : numpy array of size 'unknown x 1'
           Vector of predicted target values
           
        '''
        n = x.shape[0]
        y_hat = self.predictive_distribution(x)
        if self.learn_type == "regression":
            return y_hat[0]
        else:
            y = np.zeros(n)
            y[y_hat > 0.5] = 1
            return self._inverse_binarise(y)
        
        
    def predictive_distribution(self,x):
        '''
        Calculates parameters of predictive distribution, returns mean and variance
        of prediction.
        
        Parameters:
        -----------
        
        x: numpy array of size 'unknown x m'
           Matrix of test explanatory variables.
           
        Returns:
        --------
        
        Regression case:
                 [mu,var]: list of size 2
                     
                 mu: numpy array of size 'unknown x 1'
                     vector of means for each data point
                 var: numpy array of size 'unknown x 1'
                     vector of variances for each data point
                     
        Classification case:
                 pr: numpy vector of size 'unknown x 1'
                     Vector of probabilities
        '''
        # kernelise data if required and choose relevant features ( support vectors )
        if self.kernel is not None:
            x = SparseBayesianLearner.kernel_estimation(x,self.Xraw[self.active[1:],:],
                                                          self.kernel,
                                                          self.scaler,
                                                          self.p_order)
        else:
            x = x[:,self.active[1:]]
        # add bias term
        if self.active[0] == True:
            x    = np.concatenate((np.ones([x.shape[0],1]), x),1)
        
        # finds mean of predictive distribution
        predictive_mean  = np.dot(x, self.Wmap)
        
        # (part of variance for gaussian dist and Laplace approx to bernoulli)
        var   = np.sum( np.dot(x,self.Sigma) * x , axis = 1)
        if self.learn_type == "regression":
            predictive_var  =  1.0 / self.beta + var
            return [predictive_mean,predictive_var]
        else:
            # use probit function for approximating convolution of sigmoid and
            # normal distributon (in case of classification problem)
            ks = 1. / ( 1. + np.pi*var / 8)**0.5
            pr = sigmoid(predictive_mean * ks)
            return pr
            
            
    def _posterior_params(self,X,Y,diagA,beta, full_covariance = False):
        '''
        Calculates mean and covariance of posterior distribution of weights.
        In case of classification posterior distribution is approximated by
        Gaussian (via Laplace approximation), so it is still determined by first
        and second moments.
        
        Parameters:
        -----------
        
        X: numpy array of size 'n x m (active)'
           Matrix of active explanatory features
           
        Y: numpy array of size 'n x 1'
           Vector of explanatory variables
        
        diagA: numpy array of size 'm x 1'
           Vector of diagonal elements for precision of prior
           
        beta: float
           Precision parameter of likelihood (is used only for regression)
           
        full_covariance: bool (DEFAULT = False)
           If False returns diagonal elements of covariance matrix, otherwise
           returns whole matrix.

        Returns:
        --------
        [Wmap,SigmaDiag] : 
                   Wmap: numpy array of size 'm x 1', mean of posterior
                   SigmaDiag: numpy array of size 'm x 1', diagonal of covariance matrix
        
        '''
        R    = None 
        S    = None
        if self.learn_type=="regression":
            
            # precision parameter of posterior
            S            = beta*np.dot(X.T,X)
            np.fill_diagonal(S, np.diag(S) + diagA)
            XY           = beta*np.dot(X.T,Y)
            
            # calculate posterior mean using Cholesky decomposition
            R            = np.linalg.cholesky(S)
            Z            = solve_triangular(R,XY, check_finite = False, lower = True)
            Wmap         = solve_triangular(R.T,Z,check_finite = False, lower = False)

        else:
            
            # use variant of Newton - Raphson to get best coeffs. for logistic regr.
            f            = lambda w: cost_grad(X,Y,w,diagA)
            w_init       = np.random.random(X.shape[1])
            Wmap         = fmin_l_bfgs_b(f, x0 = w_init, pgtol   = self.pgtol_irls,
                                                       maxiter = self.max_iter_irls)[0]
            # calculate negative of Hessian at w = Wmap (for Laplace approximation)
            s            = sigmoid(np.dot(X,Wmap))
            Z            = s * (1 - s)
            S            = np.dot(X.T*Z,X)
            np.fill_diagonal(S,np.diag(S) + diagA)
            R            = np.linalg.cholesky(S)
            
        # note we do not need whole inverse only diagonal of inverse
        # in order to find diagonal elements of inverse we first invert R (chol)
        Ri           = solve_triangular(R,np.eye(self.m), check_finite = False, lower = True)
        if full_covariance is True:
            # is used only after convergence, returns full inverse of precision
            SigmaDiag    = np.dot(Ri.T,Ri)
        else:
            # is used for evidence maximization
            SigmaDiag    = np.sum(Ri**2,0)
        return [Wmap,SigmaDiag]
        
        
    def _binarise(self, Y, classes):
        '''
        Transform vector of two classes into binary vector
        '''
        self.inverse_encoding = {}
        for el,val in zip(list(classes),[0,1]):
            self.inverse_encoding[val] = el
        y  =  np.zeros(Y.shape[0])
        y[Y==self.inverse_encoding[1]] = 1
        return y
        
    
    def _inverse_binarise(self,y):
        '''
        Transform binary vector into vector of original classes
        
        Parameters:
        -----------
        y: numpy 
        '''
        Y = np.array([self.inverse_encoding[0] for e in range(y.shape[0])])
        Y[y==1] = self.inverse_encoding[1]
        return Y
        
        
    @staticmethod
    def kernel_estimation(K1,K2,kernel_type, scaler, p_order):
        '''
        Calculates value of kernel for data given in matrix K.
        
        Parameters:
        -----------
        
        K1: numpy array of size 'n1 x m'
           Matrix of variables
           
        K2: numpy array of size 'n2 x m'
           Matrix of variables (in case of prediction support vectors)
           
        kernel_type: str
           Kernel type , can be 
                                -  gaussian  exp(-(outer_sum(mu,mu') - X*X')/scaler)
                                -  poly      (c + X*X'/ scaler)^p_order
                                
        scaler: float
           value of scaling coefficient 
           
        p_order: float
           Order of polynomial ( valid for polynomial kernel)
           
        Returns:
        --------
        
        kernel: numpy array of size 'n x n' - kernel
        '''
        # inner function for distance calculation
        def dist(K1,K2):
            ''' Calculates distance between observations of matrix K'''
            n1,m1 = K1.shape
            n2,m2 = K2.shape
            # outer sum of two m x 1 matrices ( correspond to x^2 + y^2)
            K1sq  = np.outer( np.sum(K1**2, axis = 1), np.ones(n2) )
            K2sq  = np.outer( np.sum(K2**2, axis = 1), np.ones(n1) ).T
            #  correspond to 2*x*y
            K12   = 2*np.dot(K1,K2.T)
            return K1sq - K12 + K2sq 
                
        if kernel_type == "gaussian":
            distSq = dist(K1,K2)
            return np.exp(-distSq/scaler)
            
        elif kernel_type == "poly":
            return (np.dot(K1,K2.T)/scaler + 1)**p_order
                
            
#------------------------ Helper functions for RVM classification -------------------------#


def sigmoid(X):
    '''
    Evaluates sigmoid function
    '''
    return 1. / ( 1 + np.exp(-X))
    
    
def sigmoid_inverse(X):
    '''
    Evaluates function 1 - sigmoid
    (is used for numerical purposes to prevent overflow, when value of sigmoid
    is close to 1)
    '''
    return np.exp(-X)/(1. + np.exp(-X))


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
    s     = sigmoid(Xw)
    # using sigmoid inverse prevents numerical problems (if s is close to 1 
    # attempt of calculating log(1-s) will result in underflow )
    si    = sigmoid_inverse(Xw)
    cost  = np.sum( -1*np.log(s)*Y - np.log(si)*(1 - Y)) + np.sum(w*w*diagA)/2
    grad  = np.dot(X.T, s - Y) + w*diagA
    return [cost/n,grad/n]
            
 
