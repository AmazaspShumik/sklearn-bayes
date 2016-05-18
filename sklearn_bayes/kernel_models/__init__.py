'''
Sparse kernel models 

   IMPLEMENTED ALGORITHMS
   ======================
   KernelisedElasticNetRegression
   KernelisedLassoRegression
   KernelisedLogisticRegressionL1
'''


from .kernel_models import (KernelisedElasticNetRegression, KernelisedLassoRegression,
                            KernelisedLogisticRegressionL1)

__all__ = ['kernel_models']