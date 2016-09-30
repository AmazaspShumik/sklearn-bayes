# -*- coding: utf-8 -*-
"""
This module implements Type II ML Linear Regression , Variational Linear Regression

    IMPLEMENTED ALGORITHMS:
    =======================
    1. Variational Bayes Linear Regression     : VBLinearRegression
    2. Empirical Bayes Linear Regression       : EBLinearRegression
    3. Variational Bayes Logistic Regression
       with Jaakola Jordan local variational
       bound                                   : VBLogisticRegression
    4. Empirical Bayes Logistic Regression 
       with Laplace Approximation              : EBLogisticRegression

"""

from .bayes_logistic import EBLogisticRegression,VBLogisticRegression
from .bayes_linear import EBLinearRegression, VBLinearRegression


__all__ = ['EBLogisticRegression','VBLogisticRegression','EBLinearRegression',
           'VBLinearRegression']

