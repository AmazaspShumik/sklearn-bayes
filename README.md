## Bayesian Machine Learning Algorithms with scikit-learn api


### Installing & Upgrading package

    pip install https://github.com/AmazaspShumik/sklearn_bayes/archive/master.zip
    pip install --upgrade https://github.com/AmazaspShumik/sklearn_bayes/archive/master.zip

### NOTE:
I am currently updating some algorithms (RVM,HMM) and writing new ipython notebooks, updated version of package should be ready by end of June 2016. I will also add Latent Dirichlet Allocation and Ising Model to package.
   
### Algorithms

  [Linear Models](https://github.com/AmazaspShumik/sklearn-bayes/tree/master/skbayes/linear_models)
  
       Type II Maximum Likelihood Bayesian Linear Regression 
       [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/linear_models/bayesian_regression.py)
       
       Type II Maximum Likelihood Bayesian Logistic Regression (uses Laplace Approximation)  [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/linear_models/bayesian_logistic.py)
       
       Variational Bayes Linear Regression  [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/linear_models/variational_regression.py)
       
       Variational Bayes Logististic Regression (uses local variational bounds) [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/linear_models/variational_logistic.py) 
       
       
  [ARD Models](https://github.com/AmazaspShumik/sklearn-bayes/tree/master/skbayes/rvm_ard_models)
  
       Relevance Vector Regression (version 2.0) [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/rvm_ard_models/fast_rvm.py)
       
       Relevance Vector Classifier (version 2.0) [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/rvm_ard_models/fast_rvm.py)
     
       Type II Maximum Likelihood ARD Linear Regression [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/rvm_ard_models/fast_rvm.py)
       
       Type II Maximum Likelihood ARD Logistic Regression [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/rvm_ard_models/fast_rvm.py)
       
       Variational Relevance Vector Regression [code](https://github.com/AmazaspShumik/sklearn_bayes/blob/master/skbayes/rvm_ard_models/vrvm.py)
       
       Variational Relevance Vector Regression [code](https://github.com/AmazaspShumik/sklearn_bayes/blob/master/skbayes/rvm_ard_models/vrvm.py)
       
       
  [Mixture Models](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/mixture_models)
  
       Variational Bayes Gaussian Mixture Model with Automatic Model Selection [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/mixture_models/mixture.py)
       
       Variational Bayes Bernoulli Mixture Model [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/mixture_models/mixture.py)
       
       Variational Multinoulli Mixture Model [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/mixture_models/mixture.py)
       
       
  [Hidden Markov Models](https://github.com/AmazaspShumik/sklearn-bayes/tree/master/skbayes/hidden_markov_models)
  
      Variational Bayes Bernoulli Hidden Markov Model [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/hidden_markov_models/hmm.py)
  
      Variational Bayes Multinoulli Hidden Markov Model [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/hidden_markov_models/hmm.py)
  
      Variational Bayes Gaussian Hidden Markov Model [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/hidden_markov_models/hmm.py)
  

### Contributions:

There are several ways to contribute
 -- improve quality of existing code (find bugs, suggest optimization, etc.)
 -- implement machine learning algorithm (it should be bayesian; you should also provide examples & notebooks)
 -- implement new ipython notebooks with examples 




[![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/AmazaspShumik/sklearn_bayes/trend.png)](https://bitdeli.com/free "Bitdeli Badge")

