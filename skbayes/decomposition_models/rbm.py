import numpy as np
from numpy.random import multinomial
from scipy.special import expit
from scipy.misc import logsumexp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import gen_even_slices, check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted, NotFittedError
from scipy.sparse import csr_matrix, issparse


# Difference between RBM in skbayes and sklearn
# 1) skbayes implements both CD and PCD (only PCD in sklearn)
# 2) skbayes implementation has L2 penalty , which is useful for regularization
# 3) skbayes uses momentum (not simple sgd)
# 4) skbayes uses adjustable learning rate (decreases with each iteration)



class BaseRBM(BaseEstimator,TransformerMixin):
    '''
    Superclass for different types of Restricted Boltzman Machines
    '''
    
    def __init__(self,n_components, n_iter, learning_rate, momentum,
                 batch_size, l2_penalty, n_gibbs_samples, verbose):
        self.n_components     = n_components
        self.n_iter           = n_iter
        self.learning_rate    = learning_rate
        self.momentum         = momentum
        self.batch_size       = batch_size
        self.l2_penalty       = l2_penalty
        self.n_gibbs_samples  = n_gibbs_samples
        self.verbose          = verbose   
        
        
    def _ph_v(self,V):
        ''' Computes probability of hidden layer activation given visible layer'''
        return expit(safe_sparse_dot(V,self.weights_.T,dense_output=True) + self.bias_hidden_)
   
   
    def _pv_h(self,H):
        ''' Computes pdf of visible layer given hidden '''
        raise NotImplementedError

        
    def _sample_hidden(self,V):
        ''' Samples from hidden layer given visible layer '''
        return np.random.random([V.shape[0],self.n_components]) < self._ph_v(V)
        
        
    def _sample_visible(self,H):
        ''' Samples from visible layer '''
        raise NotImplementedError
   
           
    def _gibbs_sampler(self,V,k):
        '''
        Runs Gibbs Sampler for k iterations
        '''
        # Due to conditional independence properties of RBM (see Hammsley-Clifford Theorem)
        # we can run block gibbs sample (i.e. sample all hidden (or visible) neurons at once)
        n_samples, n_features = V.shape
        for i in xrange(k):
            # Sample Hidden Layer | Visible Layer
            H = self._sample_hidden(V)
            
            # Sample Visible Layer | Hidden Layer
            V = self._sample_visible(H)
            
        return V            
        
              
    def _mini_batch_compute(self, n_samples):
        ''' 
        Compute equal sized minibatches ( indexes )
        This method is taken from sklearn/neural_network/rbm.py
        '''
        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        batch_slices = list(gen_even_slices(n_batches * self.batch_size,n_batches, n_samples))
        return batch_slices   
       
        
    def _init_params(self,X):
        ''' 
        Initialise parameters, parameter initialization is done using receipts
        from [2]
        '''
        n_samples, n_features = X.shape
        self.bias_hidden_  = np.zeros(self.n_components)
        self.bias_visible_ = np.zeros(n_features) # size = (n_features,)
        self.weights_      = np.random.normal(0,0.01,(self.n_components,n_features))


    def _update_params(self,X,v):
        ''' 
        Update parameters using approximation to gradient (works with  
        contrastive divergence and persistent contrastive divergence)
        '''
        n_samples = X.shape[0]
        # P(H = 1 | V) - activation probability given sampled visible layer
        ph_v  = self._ph_v(v)
        # P(H = 1 | X) - activation probability given data
        ph_x = self._ph_v(X)
        
        # compute gradients
        grad_bias_hidden  = np.sum(ph_x,0) - np.sum(ph_v,0)
        # since input can be sparse use np.asarray(:).squeeze() to transform it
        grad_bias_visible = np.asarray(X.sum(0)).squeeze() - np.asarray(v.sum(0)).squeeze()
        grad_weights      = safe_sparse_dot(ph_x.T, X, True) - safe_sparse_dot(ph_v.T, v, True)
        
        # L2 penalty is not applied to bias terms! ( for regularization and faster mixing )
        grad_weights     -= self.l2_penalty * self.weights_
        
        # normalise gradients by sample size
        grad_bias_hidden  = grad_bias_hidden / n_samples 
        grad_bias_visible = grad_bias_visible / n_samples
        grad_weights      = grad_weights / n_samples
        
        # update learning rate (gradually decreasing it)
        # this is heuristics , we assume every 5 calls to _fit is equivalent to
        # one epoch
        lr = 1 / ( 1. / self.learning_rate + float(self.t_) / 5 )
        
        # velocity for momentum updates
        self.velocity_h_bias_  = self.momentum * self.velocity_h_bias_ + lr * grad_bias_hidden
        self.velocity_v_bias_  = self.momentum * self.velocity_v_bias_ + lr * grad_bias_visible
        self.velocity_weights_ = self.momentum * self.velocity_weights_+ lr * grad_weights
           
        # update parameters ( note we do gradient ASCENT !)            
        self.weights_      = self.weights_ + self.velocity_weights_ 
        self.bias_hidden_  = self.bias_hidden_ + self.velocity_h_bias_
        self.bias_visible_ = self.bias_visible_ + self.velocity_v_bias_        
        
        
    def _general_fit(self,X):
        ''' Fit model '''
        X = check_array(X, accept_sparse = ['csr'])
        
        # initialise paramters ( all parameter initialization is done as it is 
        # described in [2])
        self._init_params(X)
        
        # separate dataset into minibatches
        mini_batch_slices = self._mini_batch_compute(X.shape[0])
        
        for epoch in xrange(self.n_iter):
            for mini_batch in mini_batch_slices:
                self._fit(X[mini_batch])
        return self
         
    
    def _general_partial_fit(self,X):
        ''' Partial fit model '''
        X = check_array(X, accept_sparse = ['csr'])
        
        # in case nothing was fitted before, initialize params
        if not hasattr(self,'weights_'):
            self._init_params(X)
        
        # separate dataset into mini-batches
        mini_batch_slices = self._mini_batch_compute(X.shape[0])
        
        for mini_batch in mini_batch_slices:
            # is there better way for sparse matrices (slicing is too 
            # expensive for csr_matrix )
            self._fit(X[mini_batch])
            
        return self
        
        
    def _reconstruct_probs(self,X):
        ''' Perfroms following computation: X -> ph_v -> pv_h'''
        check_is_fitted(self,'weights_')
        X = check_array(X,accept_sparse=['csr'])
        H = self._ph_v(X) >= 0.5
        V = self._pv_h(H)
        return V
    

    def transform(self,X):
        '''
        Hidden Layer represenatation of observed data.
        
        Parameters
        ----------
        X: {array-like or csr_matrix} of size (n_samples, n_features)
           Data Matrix
        
        Returns
        -------
        ph_v: numpy array of size (n_samples, n_components)
           Activation probability of hidden layer
        '''
        check_is_fitted(self,'weights_')
        X = check_array(X, accept_sparse = ['csr'])  
        ph_v =  self._ph_v(X)
        return ph_v

        

#====================  Bernoulli Restricted Boltzman Machine =====================        
        
        
class BernoulliRBM(BaseRBM):
    '''
    Restricted Boltzman Machine with Bernoulli visible units
    
    Parameters
    ----------
    n_components: int 
       Number of neurons in hidden layer
       
    n_iter: int, optional (DEFAULT = 5)
       Number of iterations (relevant only in case of using fit method, ignore
       if you are using partial_fit)
       
    optimizer: string , possible values = {'cd', 'pcd'}
       Method used in gradient approximation of Restricted Boltzman Machine
       --  'cd'  : Contrastive Divergence 
       --  'pcd' : Persistent Contrastive Divergence
       
    learning_rate: float, optional (DEFAULT = 1e-2)
       Scaling factor for gradient in momentum updates. Implementation It is not advised
       to set large learning rate. Note that algorithm will automatically decrease
       learning rate used in learning (so you provide only starting point)

       v = momentum * v + f(learning_rate) * gradient
       new_parameters = old_parameters + velocity , where f(learning_rate) is
                                                    decreasing function of iterations

    momentum: float, optional (DEFAULT = 0.5, as is advised in [2]) 
       Momentum for gradient descent update. Should be between 0 and 1.

    batch_size: int, optional (DEFAULT = 100)
       Number of examples per mini-batch (It is advised to set batch_size between
       10 and 100, do not make mini-batchs too large , see [2] for details)    
    
    l2_penalty: float, optional (DEFAULT = 1e-5)
       Standard L2 penalization, helps to avoid overfitting and improves mixing
       rate for Gibbs Sampler (see [2] for details)
    
    n_gibbs_samples: int, optional (DEFAULT = 1)
       Number of iteratins that gibbs sampler runs before update of parameters

     
    References
    ----------
    [1] Training Restricted Boltzman Machines Using Approximation to Likelihood
        Gradient (Tieleman 2008)
    
    [2] A Practical Guide to Training Restricted Boltzman Machines (Hinton 2010)
        http://www.cs.toronto.edu/%7Ehinton/absps/guideTR.pdf
        
    [3] Introduction to Restricted Boltzman Machines (Fishcer & Igel 2010)
        http://image.diku.dk/igel/paper/AItRBM-proof.pdf
    
    [4] Machine Learning A Probabilistic View (Kevin Murphy 2012)
        Chapter 27
    
    '''
    
    def __init__(self, n_components, n_iter=5, optimizer = 'pcd', learning_rate= 1e-2,
                 momentum = 0.5, batch_size=100, l2_penalty = 1e-3, n_gibbs_samples = 1,
                 verbose = False):
        # initialise through superclass
        super(BernoulliRBM,self).__init__(n_components, n_iter, learning_rate, momentum,
                                          batch_size, l2_penalty, n_gibbs_samples, 
                                          verbose)
        if optimizer not in ['cd','pcd']:
            raise ValueError(( "Parameter optimizer can be either 'pcd' or "
                               "'cd', observed : {0}").format(optimizer))
        self.optimizer = optimizer


    def _pv_h(self,H):
        ''' Computes probability of visible layer activation given hiddne layer'''
        return expit(safe_sparse_dot(H,self.weights_, dense_output=True) + self.bias_visible_)


    def _sample_visible(self,H):
        ''' Samples from visible layer given hidden layer '''
        return np.random.random([H.shape[0],self.bias_visible_.shape[0]]) < self._pv_h(H)
        
     
    def _fit(self, X):
        ''' Updates model for single batch '''
        
        n_samples, n_features = X.shape
        
        # number of previous updates (used for decreasing learning rate)
        if not hasattr(self,'t_'):
            self.t_ = 0
            # initailise these variables during first call to _fit method
            self.velocity_h_bias_  = 0
            self.velocity_v_bias_  = 0
            self.velocity_weights_ = 0
        self.t_ += 1
               
        # Contrastive Divergence
        if self.optimizer == 'cd':
            v    = self._gibbs_sampler(X,self.n_gibbs_samples)
            
        # Persistent Contrastive Divergence (do not reinitialise chain after each update)
        else:
            # initialise persistent chain in case it was not done before.
            # Persistent chain is independent of training data, so we initialise
            # chain randomly and then continue running during whole fitting process
            if not hasattr(self,'persistent_visible_'):
                initial_hidden           = np.zeros([self.batch_size,self.n_components])
                self.persistent_visible_ = self._sample_visible(initial_hidden)
                                
            # continue sampling from persistent chain
            v    = self._gibbs_sampler(self.persistent_visible_, self.n_gibbs_samples)
            
            # save last sample (to continue on next iteration)
            self.persistent_visible_ = v
            
            # handle the case when one of minibatches is smaller
            if n_samples < self.persistent_visible_.shape[0]:
                v = v[np.random.randint(0,self.batch_size,n_samples),:]
        self._update_params(X,v)
        
        
    def partial_fit(self,X):
        '''
        Fit RBM to part of data. Use this method when you can not fit the whole
        dataset.
        
        Parameters
        ----------
        X: {array-like or sparse matrix} of size (n_samples, n_features)
           Data Matrix
           
        Returns
        -------
        self: object
           self (already fitted model)
        '''
        return self._general_partial_fit(X)
        
        
    def fit(self,X):
        '''
        Fit Restricted Boltzman Machines.
        
        Parameters
        ----------
        X: {array-like or sparse matrix} of size (n_samples, n_features)
           Data Matrix
           
        Returns
        -------
        self: object
           self (already fitted model)
        '''
        return self._general_fit(X)
        
        
    def reconstruct(self,X):
        '''
        Deterministic method of reconstructing input data. Data are at first 
        propagated to hidden layer and then propagated back to visible layer.
        X --> p( Hidden | Visible ) --> p(Visible | Hidden)
        
        Parameteres
        -----------
        X: {array-like or sparse matrix} of size (n_samples,n_features)
           Data Matrix
           
        Returns
        -------
        : numpy array of size (n_samples,n_features)
           Probability of activation for each visible neuron after decoding
           from hidden layer representation.
        '''
        return self._reconstruct_probs(X)
        
    
    def sample(self, X, k = 1):
        '''
        Samples from fitted RBM and returns 'imagined' visible neurons state
        
        Parameters
        ----------
        X: {array-like or sparse matrix} of size (n_samples,n_features)
           Data Matrix
           
        k: int, optional (DEFAULT = 1)
           Number of iterations for gibbs sampler
           
        Returns
        -------
        visible_imagined: numpy array of size (n_samples,n_features)
           Samples from Gibbs Sampler
        '''
        check_is_fitted(self,'weights_')
        X = check_array(X,accept_sparse = ['csr'])
        visible_imagined = self._gibbs_sampler(X,k)
        return visible_imagined
        

    
# ========================  Replicated Softmax Model =============================
    
class ReplicatedSoftmax(BaseRBM):
    '''
    Replicated Softmax Model (Undirected Topic Model)
    
    Categorical Restricted Boltzman Machine with shared weights for each 
    hidden unit.
    
    Parameters
    ----------
    n_components: int 
       Number of neurons in hidden layer (Can be interpreted as number of topics)
       
    n_iter: int, optional (DEFAULT = 5)
       Number of iterations (relevant only in case of using fit method, ignore
       if you are using partial_fit)

    learning_rate: float, optional (DEFAULT = 1e-2)
       Scaling factor for gradient in momentum updates. Implementation It is not advised
       to set large learning rate. Note that algorithm will automatically decrease
       learning rate used in learning (so you provide only starting point)

       v = momentum * v + f(learning_rate) * gradient
       new_parameters = old_parameters + velocity 

    momentum: float, optional (DEFAULT = 0.9, as is advised in [2]) 
       Momentum for gradient descent update. Should be between 0 and 1.

    batch_size: int, optional (DRFAULT = 100)
       Number of examples per mini-batch (It is advised to set batch_size between
       10 and 100, do not make mini-batchs too large , see [2] for details)    
    
    l2_penalty: float, optional (DEFAULT = 1e-5)
       Standard L2 penalization, helps to avoid overfitting and improves mixing
       rate for Gibbs Sampler (see [2] for details)
    
    n_gibbs_samples: int, optional (DEFAULT = 1)
       Number of iteratins that gibbs sampler runs before update of parameters
    
    
    References
    ----------
    [1] Replicated Softmax Model: an Undirected Topic Model 
        ( Salakhutdinov and Hinton 2010 )
        
    [2] A Practical Guide to Training Restricted Boltzman Machines (Hinton 2010)
        http://www.cs.toronto.edu/%7Ehinton/absps/guideTR.pdf
    
    '''
    
    def __init__(self,n_components, n_iter=5, learning_rate=1e-2, momentum=0.9,
                 batch_size=100, l2_penalty=1e-3, n_gibbs_samples=1, verbose=False):
        super(ReplicatedSoftmax,self).__init__(n_components, n_iter, learning_rate, momentum,
                                          batch_size, l2_penalty, n_gibbs_samples, 
                                          verbose)
    
    
    def _pv_h(self,H):
        ''' Computes conditional probability of visible layer given hidden '''
        log_p  = safe_sparse_dot(H,self.weights_,dense_output=True)+self.bias_visible_
        log_p -= logsumexp(log_p,1,keepdims=True)
        pv_h   = np.exp(log_p)
        return pv_h
    
    
    def _sample_visible(self,H):
        ''' Samples from visible layer given hidden'''
        return np.array([multinomial(w_n,pvh_n) for w_n,pvh_n in zip(self.n_words_,self._pv_h(H))])        
        
        
    def _fit(self,X):
        ''' Fit Replicated Softmax Model'''
        n_samples, n_features = X.shape
        self.n_words_ = np.asarray(X.sum(1)).squeeze()
        
        # number of previous updates (used for decreasing learning rate)
        if not hasattr(self,'t_'):
            self.t_ = 0
            # initailise these variables during first call to _fit method
            self.velocity_h_bias_  = 0
            self.velocity_v_bias_  = 0
            self.velocity_weights_ = 0
        self.t_ += 1
        v    = self._gibbs_sampler(X,self.n_gibbs_samples)
        self._update_params(X,v)
        

    def partial_fit(self,X):
        '''
        Fit Replicated Softmax Model to part of data. Use this method in case
        you can not fit the whole dataset.
        
        Parameters
        ----------
        X: {array-like or sparse matrix} of size (n_samples,n_features)
           Term frequency matrix (i.e. X[i,j] - number of times word j appears
           in document i)
           
        Returns
        -------
        self: object
           self (already fitted model)
        '''
        return self._general_partial_fit(X)
        
        
    def fit(self,X):
        '''
        Fit Replicated Softmax Model.
        
        Parameters
        ----------
        X: {array-like or sparse matrix} of size (n_samples,n_features)
           Term frequency matrix (i.e. X[i,j] - number of times word j appears
           in document i) 
           
        Returns
        -------
        self: object
           self (already fitted model)
        '''
        return self._general_fit(X)

        
    def reconstruct(self,X):
        '''
        Deterministic method of reconstructing document word matrix
        
        Parameteres
        -----------
        X: {array-like or sparse matrix} of size (n_samples,n_features)
           Term frequency matrix (i.e. X[i,j] - number of times word j appears
           in document i)
           
        Returns
        -------
        : numpy array of size (n_samples,n_features)
           Expected term frequency matrix (TF matrix obtain after reconstruction)
        '''
        probs = super(ReplicatedSoftmax,self)._reconstruct_probs(X)
        if issparse(X):
            return probs*np.asarray(X.sum(1))
        return probs * np.sum(X,1,keepdims = True)
        


if __name__ == "__main__":
    import pandas as pd
    data = np.asarray(pd.read_csv('digits.csv'))
    
    rbm = BernoulliRBM(n_components = 900, batch_size = 10, optimizer = 'cd')
    for i in xrange(20):
        x = data[np.random.randint(0,299,100),:]
        rbm.partial_fit(x)
    
    v = rbm.reconstruct(data[0:3,:])
    
    import matplotlib.pyplot as plt
    from matplotlib import cm
    import matplotlib
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize = (40,16))
    for i,ax in enumerate(axes):
        im = ax.imshow(np.reshape(v[i,:],(28,28)), vmin=0, vmax=1, cmap = cm.coolwarm)
        cax,kw = matplotlib.colorbar.make_axes(ax)
        plt.colorbar(im, cax = cax)
    plt.show()
    
    
    topic_one   = [0.05,0.05,0.05,0.05,0.4,0.4]
    topic_two   = [0.05,0.05,0.4,0.4,0.05,0.05]
    topic_three = [0.4,0.4,0.05,0.05,0.05,0.05]
    doc_sizes   = np.random.randint(100,1000,1000)
    X           = np.zeros([1000,6])
    X[0:200,:]  = multinomial(203,topic_one,200) 
    X[200:600,:]  = multinomial(410,topic_two,400) 
    X[600:1000,:]  = multinomial(48,topic_three,400)
    rsm = ReplicatedSoftmax(n_components = 10, batch_size = 10)
    
    for i in xrange(20):
        x = X[np.random.randint(0,999,100),:]
        rsm.partial_fit(x)
        
        
    print rsm.reconstruct(X[0:10,:])
    print "Topic TWO"
    print rsm.reconstruct(X[400:410,:])
    
    #===================== Example with 20 newsgroups =========================
    
    n_features = 500
    n_samples  = 10000 
    n_train    = 6000
    n_test     = n_train + 1000

    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
    data_samples = dataset.data[:n_samples]
    data_targets = dataset.target[:n_samples]
    #data_target_names = dataset.target_names[data_targets]
    

    # term frequency vectorizer                                   
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=5, stop_words = 'english',
                                    max_features = n_features)
    tf = tf_vectorizer.fit_transform(data_samples)
    
    # Fit Replicated Softmax
    rsm20 = ReplicatedSoftmax(n_components = 200, batch_size = 100, n_iter = 5)
    rsm20.fit(tf[0:n_train])
    
    # Fit Latent Dirichlet Allocation
    lda = LatentDirichletAllocation(n_topics = 200, max_iter=5,
                                    learning_method='online')
    lda.fit(tf[0:n_train,:])

    
    from sklearn.linear_model import LogisticRegression
    
    lr_rsm = LogisticRegression( C = 10 ) 
    lr     = LogisticRegression( C = 10 )
    lr_lda = LogisticRegression( C = 10 )
    
    lr.fit(tf[0:n_train,:],data_targets[0:n_train])
    lr_rsm.fit(rsm20.transform(tf[0:n_train,:]), data_targets[0:n_train])
    lr_lda.fit(lda.transform(tf[0:n_train,:]), data_targets[0:n_train])
    
    yhat = lr.predict(tf[n_train:n_test,:])
    yrsm = lr_rsm.predict(rsm20.transform(tf[n_train:n_test,:]))
    ylda = lr_lda.predict(lda.transform(tf[n_train:n_test,:]))
    
    print np.sum(yrsm!=data_targets[n_train:n_test])
    print np.sum(yhat!=data_targets[n_train:n_test])
    print np.sum(ylda!=data_targets[n_train:n_test])