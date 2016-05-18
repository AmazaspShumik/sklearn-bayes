# -*- coding: utf-8 -*-
"""
This module implements Type II ML Linear Regression , Variational Linear Regression

    IMPLEMENTED ALGORITHMS:
    =======================
    Variational Linear Regression : VariationalLinearRegression
    Type II ML Linear Regression  : BayesianRegression

"""

from .bayesian_logistic import EBLogisticRegression
from .bayesian_regression import EBLinearRegression
from .variational_logistic import VBLogisticRegression
from .variational_regression import VBLinearRegression


__all__ = ['EBLogisticRegression','VBLogisticRegression','EBLinearRegression',
           'VBLinearRegression']

