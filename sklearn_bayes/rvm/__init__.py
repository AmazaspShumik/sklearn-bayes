# -*- coding: utf-8 -*-
'''
Fast Relevance Vector Machine & ARD ( Tipping and Faul (2003) )

    IMPLEMENTED ALGORITHMS:
    =======================
    Relevance Vector Regression : RVR
    Relevance Vector Classifier : RVC
    Classification ARD          : ClassificationARD
    Regression ARD              : RegressionARD
'''
from .fast_rvm import RVR,RVC,ClassificationARD,RegressionARD

__all__ = ['RVR','RVC','ClassificationARD','RegressionARD']


