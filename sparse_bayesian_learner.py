# -*- coding: utf-8 -*-

import numpy as np

class SparseBayesianLearner(object):
    '''
    Implements Sparse Bayesian Learner, in case no kernel is given this is equivalent
    to regression with automatic relevance determination, if kernel is given it is
    equivalent to relevance vector machine (see Tipping 2001).
    
    Parameters:
    -----------
    
    X: numpy array of size 'n x m'
       Matrix of explanatory variables
       
    Y: numpy vector of size 'n x 1'
       Vector of dependent variables
       
    bias_term: bool
       
       
    alpha_max: float
       If alpha corresponding to basis vector will be above alpha_max, then basis
       vector is pruned (i.e. not used in further computations)
    
    
    '''
    
    
    def __init__(self, X, Y, bias_term = False, alpha_max = 1e+9):
        '''
        
    
    
    
    
    