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

# TODO: cross-entropy between the input and reconstruction


        
class BernoulliRBM(BaseEstimator, TransformerMixin):
    '''
    Restricted Boltzmann Machine with Bernoulli visible units
    
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
       Learning rate for gradient descent algorithm at initial stage of learning
       (learning rate is gradually decreased)

       v = momentum * v + f(learning_rate) * gradient
       new_parameters = old_parameters + velocity, where f is monotonically decreasing
                                                   function of number of iterations

    momentum: float, optional (DEFAULT = 0.5, as is advised in [2]) 
       Momentum for gradient descent update. Should be between 0 and 1.

    batch_size: int, optional (DEFAULT = 10)
       Number of examples per mini-batch (It is advised to set batch_size between
       10 and 100, do not make mini-batchs too large , see [2] for details)    
    
    l2_penalty: float, optional (DEFAULT = 1e-5)
       Standard L2 penalization, helps to avoid overfitting and improves mixing
       rate for Gibbs Sampler (see [2] for details)
    
    n_gibbs_samples: int, optional (DEFAULT = 1)
       Number of iteratins that gibbs sampler runs before update of parameters

     
    References
    ----------
    [1] Training Restricted Boltzmann Machines Using Approximation to Likelihood
        Gradient (Tieleman 2008)
    
    [2] A Practical Guide to Training Restricted Boltzmann Machines (Hinton 2010)
        http://www.cs.toronto.edu/%7Ehinton/absps/guideTR.pdf
        
    [3] Introduction to Restricted Boltzmann Machines (Fishcer & Igel 2010)
        http://image.diku.dk/igel/paper/AItRBM-proof.pdf
    
    [4] Machine Learning A Probabilistic View (Kevin Murphy 2012)
        Chapter 27
        
    [5] Restricted Boltzmann Machines
        (http://www.deeplearning.net/tutorial/rbm.html#rbm)
    
    '''
    def __init__(self, n_components, n_iter=5, optimizer = 'pcd', learning_rate= 1e-2,
                 momentum = 0.5, batch_size=10, l2_penalty = 1e-3, n_gibbs_samples = 1,
                 compute_score = False, verbose = False):
        self.n_components     = n_components
        self.n_iter           = n_iter
        self.learning_rate    = learning_rate
        self.momentum         = momentum
        self.batch_size       = batch_size
        self.l2_penalty       = l2_penalty
        self.n_gibbs_samples  = n_gibbs_samples
        self.verbose          = verbose  
        if optimizer not in ['cd','pcd']:
            raise ValueError(( "Parameter optimizer can be either 'pcd' or "
                               "'cd', observed : {0}").format(optimizer))
        self.optimizer        = optimizer
        self.scores_          = []
        self.compute_score    = compute_score
        



    def _init_params(self,X):
        ''' 
        Initialise parameters, parameter initialization is done using receipts
        from [2]
        '''
        n_samples, n_features = X.shape
        self.bias_hidden_  = np.zeros(self.n_components)
        self.bias_visible_ = np.zeros(n_features) # size = (n_features,)
        self.weights_      = np.random.normal(0,0.01,(self.n_components,n_features))



    def _ph_v(self,V):
        ''' Computes probability of hidden layer activation given visible layer'''
        return expit(safe_sparse_dot(V,self.weights_.T,dense_output=True) + self.bias_hidden_)
   
   

    def _pv_h(self,H):
        ''' Computes probability of visible layer activation given hiddne layer'''
        return expit(safe_sparse_dot(H,self.weights_, dense_output=True) + self.bias_visible_)



    def _sample_visible(self,H):
        ''' Samples from visible layer given hidden layer '''
        return np.random.random([H.shape[0],self.bias_visible_.shape[0]]) < self._pv_h(H)
     
     

    def _sample_hidden(self,V):
        ''' Samples from hidden layer given visible layer '''
        return np.random.random([V.shape[0],self.n_components]) < self._ph_v(V)
        


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
        
        
        
    def _neg_free_energy(self,V):
        ''' Compute -1 * free energy  (i.e. log p(V) * Z, where Z - normalizer) '''
        # sum_j = 1:M b_j * Vj
        fe  = safe_sparse_dot(V,self.bias_visible_,dense_output = True)
        # sum_j=1:M  log( 1 + exp(sum_i=1:N Wij * Vj))
        fe += np.log( 1 + np.exp( self.bias_hidden_ + 
                safe_sparse_dot(V,self.weights_.T))).sum(1)
        return fe
    
        
        
    def _update_params(self,X,v):
        ''' 
        Update parameters using approximation to gradient (works with  both CD-k
        and PCD-k )
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
        # this is heuristics , we assume every 2 calls to _fit is equivalent to
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

        # compute stochastic approximation to pseudo-loglikelihood
        if self.compute_score:
            self.scores_.append(self.pseudo_loglikelihood(X))
        if self.verbose:
            if self.compute_score:
                print(("[RBM] Partial Fit Completed, pseudo-loglikelihood "
                      "is {0}")).format(self.scores_[-1])
            else:
                print("[RBM] Partial Fit Completed")
            
        return self
        
        
        
    def fit(self,X):
        '''
        Fit Restricted Boltzmann Machines.
        
        Parameters
        ----------
        X: {array-like or sparse matrix} of size (n_samples, n_features)
           Data Matrix
           
        Returns
        -------
        self: object
           self (already fitted model)
        '''
        X = check_array(X, accept_sparse = ['csr'])
        
        # initialise paramters ( all parameter initialization is done as it is 
        # described in [2])
        self._init_params(X)
        
        # separate dataset into minibatches
        mini_batch_slices = self._mini_batch_compute(X.shape[0])
        
        for epoch in xrange(self.n_iter):
            for mini_batch in mini_batch_slices:
                self._fit(X[mini_batch])
             
            # compute stochastic approximation to pseudo-loglikelihood
            if self.compute_score:
                self.scores_.append(self.pseudo_loglikelihood(X))
            if self.verbose:
               if self.compute_score:
                     print("[RBM] Epoch {0}, pseudo-loglikelihood {1} ".format(epoch,
                                                                         self.scores_[-1]))
               else:
                     print("[RBM] Epoch {0}".format(epoch))
        return self
        
        
        
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
        
        
        
    def reconstruct(self,H):
        '''
        Deterministic method of reconstructing input data from latent space 
        represenatation.
        
        Parameteres
        -----------
        H: {array-like or sparse matrix} of size (n_samples,n_components)
           Latent space representation of data (i.e. after applying transform
           method)
           
        Returns
        -------
        : numpy array of size (n_samples,n_features)
           Probability of activation for each visible neuron after decoding
           from hidden layer representation.
        '''
        check_is_fitted(self,'weights_')
        H = check_array(H,accept_sparse=['csr'])
        H = (H >= 0.5) * 1
        V = self._pv_h(H)
        return V
        
        
    def pseudo_loglikelihood(self,v):
        '''
        Computes stochastic approximation to proxy of loglikelihood, see [5] for
        more details.
        
        Parameters
        ----------
        V: {array-like or sparse matrix} of size (n_samples,n_features)
           Samples from visible layer.
           
        Returns
        -------
        score: float
           Value of average pseudo-loglikelihood

        Mathematical Note
        ----------------
        Instead of computing pseudo-loglikelihood, which involves following computation
        for each visible unit v_{i}:
        
        log P(v_{i}| v_{-i}) = exp( - FE(v) ) / ( sum_v_{i} exp(-FE(v)) )
        
        We make following stochastic approximation:
        
         N * exp( - FE(v) ) / ( exp(-FE(v)) + exp(-FE(flipped(v_{i}), v_{-i})))
        
        '''
        check_is_fitted(self,'weights_')
        v = check_array(v,accept_sparse=['csr'])
        
        # number of samples, and number of visible units per sample
        n_sample, n_vis = v.shape
        # binarise Visisble
        V = (v >= 0.5) * 1 
                
        # the way to flip bits is taken from here (see score_sample method):
        # https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/neural_network/rbm.py
        
        # randomly choose parts bit that will be flipped 
        flip_index = (np.arange(V.shape[0]),
                      np.random.randint(0, V.shape[1], V.shape[0]))
        if issparse(V):
            data = -2 * V[flip_index] + 1
            V_ = V + csr_matrix((data.A.ravel(), flip_index), shape=V.shape)
        else:
            V_ = V.copy()
            V_[flip_index] = 1 - V_[flip_index]
        
        fe_original = self._neg_free_energy(V) 
        fe_flipped  = self._neg_free_energy(V_)

        # approximate sum_v_{j} exp ( - E(v,h) ), where j is index of flipped 
        # visible unit
        fe_total    = np.logaddexp(fe_original,fe_flipped)
        # pseudo-loglikelihood per sample
        pll         = n_vis * ( fe_original - fe_total )
        return  np.mean( pll )
        
        
    
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
