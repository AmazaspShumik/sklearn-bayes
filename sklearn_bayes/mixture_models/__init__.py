'''
Mixture Models
   
   IMPLEMENTED ALGORITHMS
   ======================
   Variational Bayesian Bernoulli Mixture Model
   Variational Bayesian Multinoulli Mixture Model
   Variational Bayesian Gaussian Mixture Model with Automatic Relevance Determination
   
'''

from .mixture import VBBMM,VBMMM,VBGMMARD

__all__ = ["VBBMM","VBMMM","VBGMMARD"]
