# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
       name = 'sklearn_bayes',
       version  = '0.1.0a1',
       description = "bayesian machine learning algorithms with scikit-learn api",
       url         = 'https://github.com/AmazaspShumik/sklearn_bayes',
       author      = 'Amazasp Shaumyan',
       author_email = 'amazasp.shaumyan@gmail.com',
       license      = 'MIT',
       packages=find_packages(exclude=['tests*']),
       install_requires=[
        'numpy>=1.9.2',
        'scipy>=0.15.1',
        'scikit-learn>=0.16.1'],
       packages=find_packages(exclude=['tests*']),
       test_suite='tests',
       tests_require=[
            'coverage>=3.7.1',
            'nose==1.3.7'],
       zip_safe=False,
       classifiers=[
        'Development Status :: 3 - Alpha',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7']
)
