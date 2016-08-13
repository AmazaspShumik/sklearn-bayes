## Bayesian Machine Learning Algorithms with scikit-learn api


### Installing & Upgrading package

    pip install https://github.com/AmazaspShumik/sklearn_bayes/archive/master.zip
    pip install --upgrade https://github.com/AmazaspShumik/sklearn_bayes/archive/master.zip

### NOTE:
I am currently updating some algorithms (RVM,HMM) and writing new ipython notebooks, updated version of package should be ready by the end of June 2016. I will also add Latent Dirichlet Allocation to package.

### Updates
Update 1: LDA (using collapsed Gibbs Sample) is included in package along with ipython notebooks. LDA is written in Cython, C extension is included.
Update 2: HMM code is updated. Forward-Max and Forward-Backwars algorithms are implemented in Cython, C extension is included.

### Further Work:
 - Dirichlet Process Mixture Models (Bernoulli, Poisson, Gaussian) using Variational Inference (should be finished by the end   of August).
 - Hierarchical Dirichlet Process (Stochastic Variational Inference, Variational Inference) (should be finished by the end of   August)
 - Still working on improving RVR stability (should finish it by the end of Spetember)
 - More tests

   
### Algorithms
 
* [Linear Models](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/linear_models)
     * Type II Maximum Likelihood Bayesian Linear Regression [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/linear_models/bayesian_regression.py)
     * Type II Maximum Likelihood Bayesian Logistic Regression (uses Laplace Approximation)  [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/linear_models/bayesian_logistic.py)
     * Variational Bayes Linear Regression  [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/linear_models/variational_regression.py)
     * Variational Bayes Logististic Regression (uses local variational bounds) [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/linear_models/variational_logistic.py) 
* [ARD Models](https://github.com/AmazaspShumik/sklearn-bayes/tree/master/skbayes/rvm_ard_models)
     * Relevance Vector Regression (version 2.0) [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/rvm_ard_models/fast_rvm.py)
     * Relevance Vector Classifier (version 2.0) [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/rvm_ard_models/fast_rvm.py)
     * Type II Maximum Likelihood ARD Linear Regression  [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/rvm_ard_models/fast_rvm.py)
     * Type II Maximum Likelihood ARD Logistic Regression  [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/rvm_ard_models/fast_rvm.py)
     * Variational Relevance Vector Regression [code](https://github.com/AmazaspShumik/sklearn_bayes/blob/master/skbayes/rvm_ard_models/vrvm.py)
     * Variational Relevance Vector Regression [code](https://github.com/AmazaspShumik/sklearn_bayes/blob/master/skbayes/rvm_ard_models/vrvm.py)
* [Mixture Models](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/mixture_models)
     * Variational Bayes Gaussian Mixture Model with Automatic Model Selection [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/mixture_models/mixture.py), [tutorial](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/ipython_notebooks_tutorials/mixture_models/example_gaussian_mixture_with_ard.ipynb)
     * Variational Bayes Bernoulli Mixture Model [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/mixture_models/mixture.py), [tutorial](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/ipython_notebooks_tutorials/mixture_models/example_bernoulli_mixture.ipynb)
     * Variational Multinoulli Mixture Model [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/mixture_models/mixture.py)
* [Hidden Markov Models](https://github.com/AmazaspShumik/sklearn-bayes/tree/master/skbayes/hidden_markov_models)
     * Variational Bayes Poisson Hidden Markov Model [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/hidden_markov_models/hmm.py), [demo](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/ipython_notebooks_tutorials/hidden_markov_models/examples_hmm.ipynb)
     * Variational Bayes Bernoulli Hidden Markov Model [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/hidden_markov_models/hmm.py)
     * Variational Bayes Gaussian Hidden Markov Model [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/hidden_markov_models/hmm.py), [demo](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/ipython_notebooks_tutorials/hidden_markov_models/examples_hmm.ipynb)
* [Decomposition Models](https://github.com/AmazaspShumik/sklearn-bayes/tree/master/skbayes/decomposition_models)
     * Latent Dirichlet Allocation (collapsed Gibbs Sampler) [code](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/skbayes/decomposition_models/gibbs_lda_cython.pyx), [tutorial](https://github.com/AmazaspShumik/sklearn-bayes/blob/master/ipython_notebooks_tutorials/decomposition_models/example_lda.ipynb)

### Contributions:

There are several ways to contribute (and all are welcomed)

     * improve quality of existing code (find bugs, suggest optimization, etc.)
     * implement machine learning algorithm (it should be bayesian; you should also provide examples & notebooks)
     * implement new ipython notebooks with examples 


[![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/AmazaspShumik/sklearn_bayes/trend.png)](https://bitdeli.com/free "Bitdeli Badge")

