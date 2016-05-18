# -*- coding: utf-8 -*-
'''
Fast Relevance Vector Machine & ARD ( Tipping and Faul (2003) )

    IMPLEMENTED ALGORITHMS:
    =======================
    Relevance Vector Regression : RVR
    Relevance Vector Classifier : RVC
    Classification ARD          : ClassificationARD
    Regression ARD              : RegressionARD
    Variational Regression ARD  : VariationalRegressionARD
    Variational Relevance Vector
                     Regression : VRVR
'''
from .fast_rvm import RVR,RVC,ClassificationARD,RegressionARD
from .vrvm import VariationalRegressionARD, VRVR

__all__ = ['RVR','RVC','ClassificationARD','RegressionARD','VariationalRegressionARD',
           'VRVR']


