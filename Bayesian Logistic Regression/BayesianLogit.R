


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
    # 3) Murphy 2012, Machine Learning A Probabilistic Perspective
    # 4) Barber 2015, Bayesian Reasoning and Machine Learning ()
    
    # ----------------- set name of the class ------------------------
    
    "BayesianLogisticRegression",
    
    # -----------------  define instance variables -------------------
    
    slots = list(
                  bias.term   = 'logical',
                  max.iter    = 'numeric',
                  conv.thresh = 'numeric',
                  scale       = 'logical',
                  muX         = 'numeric', # vector of column means
                  sdX         = 'numeric', # vector of column standard devs
                  w.mean0     = 'numeric', 
                  w.prec0     = 'numeric',
                  coefs       = 'matrix',
                  coefs.cov   = 'matrix',
                  N           = 'numeric',
                  M           = 'numeric'
                 ),
                 
    # ------------- default values for instance variables -------------
    
    prototype = list(
                      bias.term   = TRUE,
                      max.iter    = 100,
                      conv.thresh = 1e-5,
                      scale       = TRUE,
                      w.prec0     = 1e-3,
                      N           = 0,
                      M           = 0
                     ),
                     
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
    setGeneric( 'fit', def = function(theObject,X,Y){ standardGeneric('fit') } )
    setMethod('fit',signature = c('BayesianLogisticRegression','matrix','numeric'), 
              definition = function(theObject,X,Y){
              	
              	# check whether dimensionality is correct, if wrong change it
              	if ( dim(X)[1]  !=  theObject@N ) theObject@N = dim(X)[1]
              	if ( dim(X)[2]  !=  theObject@M ) theObject@M = dim(X)[2]
              	N               =   theObject@N
              	
              	# transform Y into matrix to have conformable arguments for inner product 
                Y = matrix(Y, ncol = 1)
              	
              	# scale X for better convergence if necessary
              	if ( theObject@scale ) {
              		theObject@muX = colMeans(X)
              		theObject@sdX = sapply(1:theObject@M, function(i){sd(X[,i])})
              		X = scale(X, center = theObject@muX, scale = theObject@sdX)
              	}
              	
              	# add bias term to matrix of explanatory variables if required
              	if ( theObject@bias.term) {
              		theObject@M          = theObject@M + 1
             		newX                 = matrix(data = NA, ncol = theObject@M, nrow = N)
              		newX[,1]             = rep(1,times = N)
              		newX[,2:theObject@M] = X
              		X                    = newX
              	}
              	M               = theObject@M
              	
              	# mean , precision and variational parameters
              	w.mean0        = rep(0, times = M)
              	alpha          = theObject@w.prec0
              	eps            = runif(N)
              	
              	# precompute some values before
              	XY   = matrix( t(X) %*% ( Y - 0.5 ) , ncol = 1)
              	
              	# iterations of EM algorithm
              	for( i in 1:theObject@max.iter){
              		
              		# E-step : find parameters of posterior distribution of coefficients
              		#          1) covariance of posterior
              		#          2) mean of posterior
              		
              		# covariance update
              		Xw        = X * matrix( rep(lambda(eps),times = M), ncol = M)   
              		XXw       = t(X) %*% Xw
              		# do not 'regularise' constant !!!
              		Diag      = diag(alpha, nrow = M)
              		#Diag[1,1] = 0
              		Sn.inv    = 2*XXw + Diag
              		
              		# Sn.inv is positive definite due to adding of positive diagonal
              		# hence Cholesky decomposition (that is used for inversion) should be stable
              		Sn       = qr.solve(Sn.inv, tol = 1e-7)
              		
              		# mean update
              		print ("XY,Sn,Sn.inv")
              		print (XY)
              		print (Sn)
              		print (Sn.inv)
              		Mn       = Sn %*% XY
              		
              		# M-step : 1) update variational parameter eps for each observation
              		#          2) update precision parameter (alpha)
              		
              		# variational parameter update
              		Xm   = (X %*% Mn)^2
              		XSX  = rowSums(X %*% Sn * X)
              		eps  = sqrt( Xm + XSX ) 
              		print("EPS,XSX,XM")
              		print (Xm)
              		print (XSX)
              		print (eps)
              		
              		# update of precision parameter for coefficients (except for )
              		alpha = M / ( sum(Mn[2:M]^2) + sum(diag((Sn))) - Sn[1,1] )
              		
              		print(Mn)
              		# check convergence
              		if ( i== theObject@max.iter-1){
              			theObject@coefs     = Mn
              			theObject@coefs.cov = Sn
              		}
              	}

              })


    # @Method Name : predict
    #
    # @Description : predicts target value for explanatory variables
    #
    # @Parameters  :
    # ==============
    #
    # X: matrix of size  (number of samples in test set, number of features)
    #    Matrix of explanatory variables
    #
    setGeneric('predict.probs', def = function(theObject,X){ standardGeneric('predict.probs')})
    setMethod('predict.probs', signature = c('BayesianLogisticRegression','matrix'), 
              definition = function(theObject,X){
              	
              	# dimensionality of X
              	n = dim(X)[1]
              	m = dim(X)[2]
              	
              	# scale test data if required
              	if ( theObject@scale ){
              		X = scale(X, center = theObject@muX, scale = theObject@sdX)
              	}
              	
              	# add bias term if required
              	if ( theObject@bias.term ){
              		newX                 = matrix(data = NA, ncol = m, nrow = n)
              		newX[,1]             = rep(1,times = n)
              		newX[,2:theObject@M] = X
              		X                    = newX
              	}
              	
              	probs = sigmoid( X %*% theObject@Mn )
              	
              	return (probs)
              })


