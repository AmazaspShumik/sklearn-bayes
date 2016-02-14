# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
version = sklearn_bayes.__version__

setup(
       name = 'sklearn_bayes',
       version  = version,
       description = "bayesian machine learning algorithms with scikit-learn api",
       url         = 'https://github.com/AmazaspShumik/sklearn_bayes',
       author      = 'Amazasp Shaumyan',
       author_email = 'amazasp.shaumyan@gmail.com',
       license      = 'MIT',
       packages=find_packages(exclude=['tests*']),
       test_suite='tests',
       tests_require=[
            'coverage>=3.7.1',
            'nose==1.3.7'],
       classifiers=[
        'Development Status :: 3 - Alpha',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7']
)
