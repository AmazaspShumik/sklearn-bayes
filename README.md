# Bayesian Regression Models

## Bayesian Regression with evidence approximation

This model is similar to ridge regression but does not require cross-validation. Works well in case of multicollinearity.
You can find python implementation [here](https://github.com/AmazaspShumik/Bayesian-Regression-Methods/blob/master/bayesian_regression.py) Matlab implementation [here](https://github.com/AmazaspShumik/Bayesian-Regression-Methods/blob/master/BayesianRegression.m). There is also small tutorial that compares OLS and Bayesian Regression [here](https://github.com/AmazaspShumik/Bayesian-Regression-Methods/blob/master/bayesian_regression_demo.ipynb).


## ARD regression and Relevance Vector Machine

 ARD regression is almost identical to Bayesian Regression, the only difference is prior on parameters. Prior used in ARD   
 results in sparse solutions, this can be considered bayesian analog of lasso regression.
 Kernelised version of ARD is called Relevance Vector Machine (RVM), it is mainly used for nonlinear regression. In many cases  RVM produces solutions comparable to SVM with only a fraction of support vectors used in SVM.

 Python code for RVM can be found  [here](https://github.com/AmazaspShumik/Bayesian-Regression-Methods/blob/master/sparse_bayesian_learner.py) and demo is  [here](https://github.com/AmazaspShumik/Bayesian-Regression-Methods/blob/master/ard_rvm_demo.ipynb)







