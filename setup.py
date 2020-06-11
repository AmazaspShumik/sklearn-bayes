# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy as np

ext_modules = cythonize('skbayes/decomposition_models/gibbs_lda_cython.pyx') + \
                  cythonize('skbayes/hidden_markov_models/hmm.pyx')

# Note: cythonize(include_path) doesn't work, see https://github.com/cython/cython/issues/1480
for ext in ext_modules:
    ext.include_dirs.append(np.get_include())

setup(
    name = 'skbayes',
    version = '0.1.0a1',
    description = "bayesian machine learning algorithms with scikit-learn api",
    url = 'https://github.com/AmazaspShumik/sklearn-bayes',
    author = 'Amazasp Shaumyan',
    author_email = 'amazasp.shaumyan@gmail.com',
    license = 'MIT',
    packages = find_packages(exclude=['tests*']),
    install_requires = [
        'numpy>=1.9.2',
        'scipy>=0.15.1',
        'scikit-learn>=0.17',
        'cython>=0.24',
        'six>=1.13.0,<2.0.0',
    ],
    test_suite = 'tests',
    tests_require = [
        'coverage>=3.7.1',
        'nose==1.3.7',
    ],
    classifiers = [
        'Development Status :: 3 - Alpha',
        'Operating System :: Mac OS X',
        'Programming Language :: Python :: 2.7',
    ],
    ext_modules = ext_modules,
)
