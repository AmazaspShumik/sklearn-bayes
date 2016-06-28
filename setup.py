# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

try:
    from numpy.distutils.misc_util import get_info
except ImportError:
    def get_info(name):
        return {}

ext_modules = [ Extension("skbayes.decomposition_models.gibbs_lda_cython",
                          ["skbayes/decomposition_models/gibbs_lda_cython.c"], 
                          #include_dirs = [np.get_include()],
                          extra_compile_args=["-O3"],
                          **get_info("npymath")),
                Extension("skbayes.hidden_markov_models.hmm",
                         ['skbayes/hidden_markov_models/hmm.c'],
                          extra_compile_args=["-O3"],
                          **get_info("npymath"))
              ]


import skbayes

 
if __name__=='__main__':           
   setup(
       name = 'skbayes',
       version  = '0.1.0a1',
       description = "bayesian machine learning algorithms with scikit-learn api",
       url         = 'https://github.com/AmazaspShumik/sklearn-bayes',
       author      = 'Amazasp Shaumyan',
       author_email = 'amazasp.shaumyan@gmail.com',
       license      = 'MIT',
       packages=find_packages(exclude=['tests*']),
       install_requires=[
          'numpy>=1.9.2',
          'scipy>=0.15.1',
          'scikit-learn>=0.17',
          'cython>=0.24'],
       test_suite='tests',
       tests_require=[
          'coverage>=3.7.1',
          'nose==1.3.7'],
       classifiers=[
          'Development Status :: 3 - Alpha',
          'Operating System :: Mac OS X',
          'Programming Language :: Python :: 2.7'],
       ext_modules = ext_modules
   )

