# Bayesian logistic regression with local variational bounds
#
# Similar to standard Bayesian Logistic Regression, but uses local 
# Jaakola-Jordan variational bound instead of Laplace approximation
#
# References:
# 1) Bishop 2006, Pattern Recognition and Machine Learning ( Chapter 10 )
# 2) Jaakola & Jordan 1994, 
# 3) Murphy 2012,
# 4) Barber 2015, Bayesian Reasoning and Machine Learning ()


# ========================== Helper Functions ==================================

sigmoid <- function(x) {
	# Calculates value of sigmoid function
	return ( 1.0 / ( 1 + exp(-x)) )
}

lambda <- function(x){
	# helper function used for local variational bound calculation
	-0.5 / x * ( sigmoid(x) - 0.5 )
}

% ======================= Bayesian Logistic Regression =========================


BayesianLogisticRegression <- setClass(
    
    # ----------------- set name of the class ------------------------
    
    "BayesianLogisticRegression",
    
    # -----------------  define instance variables -------------------
    
    slots = list(
                  bias.term   = 'logical',
                  max.iter    = 'numeric',
                  conv.thresh = 'numeric',
                  w.mean0     = 'numeric',
                  w.prec0     = 'numeric',
                  N           = 'numeric',
                  M           = 'numeric'
                 )
                 
    # ------------- default values for instance variables -------------
    
    prototype = list(
                      bias.term   = TRUE,
                      max.iter    = 100,
                      conv.thresh = 1e-5
                      w.prec0     = 1e-6
                      w.mean0     = 0
                      N           = 0
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
              definition = function(theObject,X,y){
              	
              	# check whether dimensionality is correct, if wrong change it
              	if ( dim(X)[1] == theObject.N ) theObject.N = dim(X)[1]
              	if ( dim(X)[2] == theObject.M ) theObject.M = dim(X)[2]
              	
              	# add bias term to matrix of explanatory variables if required
              	if ( theObject@bias.term) X = 
              	
              	
              	for( i in 1:theObject@max.iter){
              		
              		
              		
              		
              		
              		
              	}
              	
              	
              	
              	
              	
              })
              
              
              
    setMethod('predict')
    setMethod('predict.probs')
    setMethod('show')
 )





