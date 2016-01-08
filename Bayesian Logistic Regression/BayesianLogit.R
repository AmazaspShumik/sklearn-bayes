


# ========================== Helper Functions ==================================

sigmoid <- function(x) {
	# Calculates value of sigmoid function
	return ( 1.0 / ( 1 + exp(-x)) )
}

lambda <- function(x){
	# helper function used for local variational bound calculation
	return (-0.5 / x * ( sigmoid(x) - 0.5 ) )
}

% ======================= Bayesian Logistic Regression =========================


BayesianLogisticRegression <- setClass(
    # Bayesian logistic regression with local variational bounds.
    # Similar to standard Bayesian Logistic Regression, but uses local 
    # Jaakola-Jordan variational bound instead of Laplace approximation
    #
    # @Parameters:
    # ============
    # bias.term  : logical vector of size (1,1) [ DEFAULT = TRUE ]
    #              If True adds columns of ones to matrix of explanatory variables
    #
    # max.iter   : numeric vector of size (1,1)  [ DEFAULT = 100 ]
    #              Maximum number of iterations before convergence
    #
    # conv.thresh: numeric vector of size (1,1) [ DEFAULT = 1e-3 ]
    #              Threshold for convergence of algorithm
    #              
    # w.mean0    : numeric vector of size ( number of features, 1)
    #              Mean of prior distribution for coefficients
    #
    # w.prec0    : matrix of size (number of features, number of features)
    #              Precision of prior distribution for coefficients
    #
    #           
    # @References:
    # ============
    # 1) Bishop 2006, Pattern Recognition and Machine Learning ( Chapter 10 )
    # 2) Jaakola & Jordan 1994, 
    # 3) Murphy 2012,
    # 4) Barber 2015, Bayesian Reasoning and Machine Learning ()
    
    # ----------------- set name of the class ------------------------
    
    "BayesianLogisticRegression",
    
    # -----------------  define instance variables -------------------
    
    slots = list(
                  bias.term   = 'logical',
                  max.iter    = 'numeric',
                  conv.thresh = 'numeric',
                  w.mean0     = 'numeric',
                  w.prec0     = 'numeric',
                  coefs       = 'numeric',
                  coefs.prec  = 'matrix',
                  N           = 'numeric',
                  M           = 'numeric'
                 )
                 
    # ------------- default values for instance variables -------------
    
    prototype = list(
                      bias.term   = TRUE,
                      max.iter    = 100,
                      conv.thresh = 1e-5,
                      w.prec0     = 1e-6,
                      N           = 0,
                      M           = 0
                     )
                             
    # ----------------------- define methods --------------------------
    
    
    # @Method Name : fit
    # 
    # @Description : Fits Bayesian Logistic Regression
    #
    # @Parameters  : 
    # ==============
    # X: matrix of dimensionality (number of samples, number of features)
    #     Matrix of explanatory variables
    #
    # Y: numeric vector of dimensionality (number of samples, 1)
    #     Vector of dependent variables
    #
    #      
    setMethod('fit',signature = c('BayesianLogisticRegression','matrix','numeric'), 
              definition = function(theObject,X,Y){
              	
              	# check whether dimensionality is correct, if wrong change it
              	if ( dim(X)[1] == theObject@N ) theObject@N = dim(X)[1]
              	if ( dim(X)[2] == theObject@M ) theObject@M = dim(X)[2]
              	N              =  theObject@N
              	M              =  theObject@M
              	
              	# add bias term to matrix of explanatory variables if required
              	if ( theObject@bias.term) {
              		newX       = matrix(data = NA, ncol = M + 1, nrow = N)
              		newX[,1]   = rep(1,times = N)
              		newX[,2:M] = X
              		X          = newX
              	}
              	
              	# mean , precision and variational parameters
              	w.mean0        = rep(0, times = M)
              	w.prec0        = theObject@w.prec0
              	eps            = runif(N)
              	
              	# precompute some values before
              	XY   = X %*% t( Y - 0.5 )
              	
              	# iterations for optimization
              	for( i in 1:theObject@max.iter){
              		
              		Sn = lambda(eps)
              		
              		
              		
              		
              		
              		
              	}
              	
              	
              	
              	
              	
              })
              
              
              
    setMethod('predict')
    setMethod('predict.probs')
    setMethod('show')
 )





