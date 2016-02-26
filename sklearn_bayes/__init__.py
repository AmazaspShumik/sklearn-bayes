'''
Bayesian machine learning models with sklearn api
=================================================

   IMPLEMENTED ALGORITHMS:
   -----------------------
   
       ** Regression 
          - Relevance Vector Regression (version 2.0)
          - Variational Relevance Vector Regression
          - Type II ML Bayesian Linear Regression
          - Variational Linear Regression
          - Type II ML ARD Regression
          - Variational ARD Regression
          
       ** Classification   
          - Relevance Vector Classifier
          - Type II ML Bayesian Logistic Regression (Laplace Approximation)
          - Variational Logistic Regression (Local Variational Approximation)
          - Type II ML ARD Classification
          
       ** Mixture Models
          - Variational Bayesian Bernoulli Mixture Model (VBBMM)
          - Variational Bayesian Multinomial Mixture Model (VBMMM)
          - Variational Bayesian Gaussian Mixture with Automatic Relevance Determination
          
          
    PACKAGE CONTENTS:
    -----------------
    
        linear (package)
        logistic (package)
        rvm (package)
        vrvm (package)
        mixture (package)

'''

__all__ = ['rvm','linear','logistic','vrvm','mixture']

__version__ = '0.1.0a2'

