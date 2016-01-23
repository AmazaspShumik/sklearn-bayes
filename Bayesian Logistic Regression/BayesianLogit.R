


# ========================== Helper Functions ==================================

sigmoid <- function(x) {
	# Calculates value of sigmoid function
	return ( 1.0 / ( 1 + exp(-x)) )
}

lambda <- function(eps){
	# helper function used for local variational bound calculation
	return (0.5 / eps * ( sigmoid(eps) - 0.5 ) )
}


# @Description: Checks whether X and Y are acceptable inputs
#
# @Parameters  : 
# ==============
# X: matrix of dimensionality (number of samples, number of features)
#     Matrix of explanatory variables
#
# Y: numeric vector of dimensionality (number of samples, 1)
#     Vector of dependent variables (it should contain only zeros and ones)
#
check_X_y <- function(X,Y){
    # make dependent variable a factor & check that it has exactly two levels
    y  =  factor(Y)
    if ( length( levels( y ) ) != 2) stop('There can be only two classes')
               	
    # check that data matrix is numeric
    if ( !is.numeric(X) ) stop('X should be numeric')
    
    # check that both X and Y have the same number of rows
    if( dim(X)[1] != length(Y) ) stop('Number of samples in X and Y should be the same')
}


# @Description : Transforms vector Y into numeric vector of zeros and ones
#
# @Parameters:
# ===========
# Y: vector of size (number of samples, 1)
#     Vector of dependent variables
#
# y: factor with 2 levels
#    Factor
# 
# @Returns:
# ========
# binariser.list:
#                 $zero      : character or numeric , corresponding to 0's in binarised vector
#                 $ones      : character or numeric , corresponding to 1's in binarised vector
#                 $numericY  : binarised vector
#
binarise <- function(Y,y){
	y_hat = (Y == levels(y)[1])*1
	binariser.list = list(zero = levels(y)[2], ones   = levels(y)[1] , 
	                      numericY = y_hat   , classY = class(Y))
	return ( binariser.list )
}


# @Description : Transforms vector binarised vector into target vector
#
# Parameters:
# ===========
# y_pred: vector of size (number of samples, 1)
#     Binarised vector (i.e. contains only 0's and 1's)
#
# binarise.list: list( zero, ones, numericY )
#     Output of binarise function (see description in binarise function)
# 
# Returns:
# ========
# Y: vector of size (number of samples, 1) 
#    Vector of target values
#
inverse.binarise <- function(y_pred, binariser.list){
	Y            = rep(binariser.list$zero, times = length(y_pred))
	Y[y_pred==1] = binariser.list$ones
	if(binariser.list$classY=='numeric') Y = as.numeric(Y)
	return (Y)
}


% =========================== Bayesian Logistic Regression ============================


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
                  M           = 'numeric',
                  binariser   = 'list'
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
                             
    # ---------------------------------- define methods --------------------------------------
    
    
    # @Method Name : fit.model
    # 
    # @Description : Fits Bayesian Logistic Regression
    #
    # @Parameters  : 
    # ==============
    # X: matrix of dimensionality (number of samples, number of features)
    #     Matrix of explanatory variables
    #
    # Y: numeric vector of dimensionality (number of samples, 1)
    #     Vector of dependent variables (it should contain only zeros and ones)
    #
    setGeneric( 'fit.model', def = function(theObject,X,Y){ standardGeneric('fit.model') } )
    setMethod('fit.model',signature = c('BayesianLogisticRegression','matrix','numeric'), 
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
              	eps            = rep(1,times = N)
              	
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
              		Mn       = Sn %*% XY
              		
              		# M-step : 1) update variational parameter eps for each observation
              		#          2) update precision parameter (alpha)
              		
              		# variational parameter update
              		Xm   = (X %*% Mn)^2
              		#XSX  = rowSums(X %*% Sn * X)
              		XSX  = diag(X %*% Sn %*% t(X))
              		eps  = sqrt( Xm + XSX ) 
              		print("EPS,XSX,XM")
              		print (det(Sn))
              		
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
              
            
              
    # @Method Name : fit
    #
    # @Description : Wrappers for fit.model method. 'fit' is overloaded method, there are 
    #                several implementations corresponding to several signatures
    #
    setGeneric( 'fit', def = function(theObject,X,Y){ standardGeneric('fit') } )
    
    
    
    # @Overloaded fit, Implementation 1
    #
    # @Parameters:
    # ============
    # X: matrix of dimensionality (number of samples, number of features)
    #     Matrix of explanatory variables
    #
    # Y: character vector of dimensionality (number of samples, 1)
    #     Vector of dependent variables 
    #
    setMethod('fit', signature = c('BayesianLogisticRegression','matrix','character'),
               definition = function(theObject,X,Y){
               	
                # check whether X and Y are acceptable inputs
                check_X_y(X,Y)
               	
               	# binarise dependent variable & save binarization data for later use 
               	# in 'predict' method
               	theObject@binariser = label.binariser(Y,y)
               	fit.model( theObject, X, theObject@binariser$numericY )
               })
               
               
               
    # @Overloaded fit, Implementation 2
    #
    # @Parameters :
    # =============
    # X: matrix of dimensionality (number of samples, number of features)
    #     Matrix of explanatory variables
    #
    # Y: numeric vector of dimensionality (number of samples, 1)
    #     Vector of dependent variables
    #
    setMethod('fit', signature = c('BayesianLogisticRegression','matrix','numeric'),
               definition = function(theObject,X,Y){
               	
                # check whether X and Y are acceptable inputs
                check_X_y(X,Y)
               	
               	# binarise dependent variable & save binarization data for later use 
               	# in 'predict' method
               	theObject@binariser = label.binariser(Y,y)
               	fit.model( theObject, X, theObject@binariser$numericY )
               })
               
               
               
    # @Overloaded fit, Implementation 3
    #
    # @Parameters :
    # =============
    # X: data.frame of dimensionality (number of samples, number of features)
    #     data.frame of explanatory variables
    #
    # Y: character vector of dimensionality (number of samples, 1)
    #     Vector of dependent variables 
    #               
    setMethod('fit', signature = c('BayesianLogisticRegression','data.frame','character'),
               definition = function(theObject,X,Y){
               	fit(theObject,as.matrix(X),Y)
               })
               
               
               
    # @Overloaded fit, Implementation 4
    #
    # @Parameters :
    # =============
    # X: data.frame of dimensionality (number of samples, number of features)
    #     data.frame of explanatory variables
    #
    # Y: numeric vector of dimensionality (number of samples, 1)
    #     Vector of dependent variables 
    #               
    setMethod('fit', signature = c('BayesianLogisticRegression','data.frame','numeric'),
              definition = function(theObject,X,Y){
              	fit(theOBject,as.matrix(X),Y)
              })
               	
               	

    # @Method Name : predict.probs
    #
    # @Description : predicts target value for explanatory variables, uses probit function
    #                for approximating convolution of sigmoid and gaussian.
    #
    setGeneric('predict.probs', def = function(theObject,X){ standardGeneric('predict.probs')})
    
    
    
    # @Overloaded predict.probs, Implementation 1
    #
    # @Parameters  :
    # ==============
    # X: matrix of size  (number of samples in test set, number of features)
    #    Matrix of explanatory variables
    #
    # @Returns:
    # =========
    #
    #
    #
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
              
              
    # @Overloaded predict.probs, Implementation 2
    #
    # @Parameters  :
    # ==============
    # X: data.frame of size  (number of samples in test set, number of features)
    #    data.frame of explanatory variables
    #
    # @Returns:
    # =========
    #
    #
    #
    setMethod('predict.probs', signature = c('BayesianLogisticRegression','matrix'), 
              definition = function(theObject,X){
              	y.hat = predict.probs(theObject,as.matrix(X))
              	return (y.hat)
             })
              
              
              
     setMethod('predict', signature = c('BayesianLogisticRegression','matrix'),
               definition = function(theObject,X){
               	
               	
               	
               	
               	
               })


