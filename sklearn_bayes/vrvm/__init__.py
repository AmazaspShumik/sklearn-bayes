# -*- coding: utf-8 -*-
"""
Variational Relevance Vector Machine & ARD ( Tipping (2001) )

    IMPLEMENTED ALGORITHMS:
    =======================
    Variational Relevance Vector Regression : VRVR
    Variational Classification ARD          : VariationalClassificationARD
    Variational Regression ARD              : VariationalRegressionARD
"""

from .vrvm import VRVR,VariationalRegressionARD

__all__=['VRVR','VariationalRegressionARD']
