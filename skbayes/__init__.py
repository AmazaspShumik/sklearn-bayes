'''
Bayesian machine learning models with sklearn api
=================================================

   IMPLEMENTED ALGORITHMS:
   -----------------------
   
       ** Linear Models
          - Type II ML Bayesian Logistic Regression with Laplace Approximation (EBLogisticRegression)
          - Type II ML Bayesian Linear Regression (EBLinearRegression)
          - Variational Bayes Linear Regression (VBLinearRegression)
          - Variational Bayes Logistic Regression (VBLogisticRegression)

       ** RVM & ARD Models   
          - Relevance Vector Regression (RVR)
          - Relevance Vector Classifier (RVC)
          - Variational Relevance Vector Regression (VRVR)
          - Variational ARD Regression (VariationalRegressionARD)
          - Type II ML ARD Regression (RegressionARD)
          - Type II ML ARD Classification (ClassificationARD)

       ** Mixture Models
          - Variational Bayesian Bernoulli Mixture Model (VBBMM)
          - Variational Bayesian Multinomial Mixture Model (VBMMM)
          - Variational Bayesian Gaussian Mixture with Automatic Relevance Determination (VBGMMARD)
          
       ** Hidden Markov Models
          - Variational Bayesian HMM with Bernoulli emission probabilities (VBBernoulliHMM)
          - Variational Bayesian HMM with Multinoulli emission probabilities (VBMultinoulliHMM)
          - Variational Bayesian HMM with Gaussian emission probabilities (VBGaussianHMM)
          
       ** Sparse Kernel Models
          - Kernelised Elastic Net Regression (KernelisedElasticNetRegression)
          - Kernelised Lasso Regression (KernelisedLassoRegression)
          - Kernelised L1 Logistic Regression (KernelisedLogisticRegressionL1)
          
       ** Decomposition Models
          - Latent Dirichlet Allocation (collapsed Gibbs Sampler)
          
          
    PACKAGE CONTENTS:
    -----------------
        linear_models (package)
        kernel_models (package)
        rvm_ard_models (package)
        mixture_models (package)
        hidden_markov_models (package)

'''

__all__ = ['rvm_ard_models','linear_models','hidden_markov_models','mixture_models',
           'kernel_models','datasets','decomposition_models']

__version__ = '0.1.0a1'

