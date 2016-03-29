from scipy.special import psi,gammaln
from scipy.misc import logsumexp
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from copy import deepcopy
import numpy as np



def _normalise(M):
    ''' Make matrix or vector stochastic (i.e. normalise by row to 1)'''
    if len(M.shape) == 1:
        return M / np.sum(M)
    return M / np.sum(M, axis = 1)
    
    
def _get_chain(X,index = []):
    ''' Generates separate chains'''
    from_idx = 0
    for idx in index:
        yield X[from_idx:idx,:]
        from_idx = idx
    if from_idx != X.shape[0]-1:
        yield X[from_idx:(X.shape[0]-1),:]

        

class VBHMM(BaseEstimator):
    '''
    Superclass for implementation of Variational Bayesian Hidden Markov 
    Models.
    
    This class implements inference steps that does not depend on form 
    of emission probabilities.
    '''
    
    def __init__(self, n_hidden = 2, n_iter = 100, init_params = None, tol = 1e-3,
                 alpha_start = 1, alpha_trans = 1, verbose = False):
        self.n_hidden    = n_hidden
        self.n_iter      = n_iter
        self.init_params = init_params
        self.tol         = tol
        self.verbose     = verbose
        self.alpha_start = alpha_start
        self.alpha_trans = alpha_trans
        
        
    
    def _init_params(self):
        ''' 
        Reads user defined parameters, or in case they are not defined randomly
        initialise
        '''
        # initial distribution
        if 'initial' in self.init_params:
            pr_start = self.init_params['initial']
        else:
            pr_start = np.random.random(self.n_hidden) * self.alpha_start
            
        # matrix of transition probabilities
        if 'transition' in self.init_params:
            pr_trans = self.init_params['transition']
        else:
            pr_trans = np.random.random( [self.n_hidden, self.n_hidden] ) * self.alpha_trans
            
        return pr_start, pr_trans
        
    
    def _probs_params(self, start_params, trans_params, emission_params, X):
        '''
        Calculate probabilities : emission, initial, transition using parameters
        '''
        log_pr_start = psi(start_params) - psi(np.sum(start_params))
        log_pr_start-= logsumexp(log_pr_start)
        log_pr_trans = psi(trans_params) - psi(np.sum(trans_params,1))
        log_pr_trans-= logsumexp(log_pr_trans,1,keepdims = True)
        pr_x         = self._emission_probs_params(emission_params,X)
        return np.exp(log_pr_start), np.exp(log_pr_trans), pr_x
        
                    
    def _fit(self, X, chain_indices = []):
        '''
        Fits Hidden Markov Model
        '''
        n_samples, n_features = X.shape

        # initialise parameters (log-scale!!!)
        start_params, trans_params, emission_params = self._init_params(n_features)
        trans_params_prior = np.copy(trans_params)
        start_params_prior = np.copy(start_params)
        emission_params_prior = deepcopy(emission_params)
        
        
        for i in range(self.n_iter):
            # statistics that accumulate data for m-step
            sf_stats = self._init_suff_stats(n_features)
            trans = np.zeros([self.n_hidden, self.n_hidden])
            start = np.zeros(self.n_hidden)
            
            # probabilies for initialised parameters           
            pr_start, pr_trans, pr_x = self._probs_params(start_params, trans_params,
                                                          emission_params, X)
                                                          
            for zx in _get_chain(X,chain_indices):
                
                print zx.shape
                alpha = self._forward_single_chain( pr_start, pr_trans, pr_x)
                trans, start, sf_stats = self._vbe_step_single_chain(zx,alpha,pr_trans,
                                                          pr_x,sf_stats, trans, start)
                
            # log parameters of posterior distributions of parameters
            trans_params, start_params, emission_params = self._vbm_step(trans,start,
                                                          sf_stats, emission_params,
                                                          trans_params_prior,
                                                          emission_params_prior,
                                                          start_params_prior) 
            print "Iteration {0}".format(i)
            if self._check_convergence():
                break
                
        return start_params, trans_params, emission_params
                
                
        
    def _vbm_step(self, trans, start, sf_stats, emission_params, trans_params_prior,
                  emission_params_prior, start_params_prior):
        '''
        Computes approximating distribution for posterior of parameters
        '''
        trans += trans_params_prior
        start += start_params_prior
        emission_params = self._vbm_emission_params(emission_params_prior, emission_params,
                                                    sf_stats)
        return trans, start, emission_params
        
        

    def _vbe_step_single_chain(self, X, alpha, pr_trans, pr_x, suff_stats, trans, start):
        '''
        Performs backward pass, at the same time computes marginal & joint marginal
        and updates sufficient statistics for VBM step
        '''
        beta_before   = np.ones(self.n_hidden)
        n_samples     = X.shape[0]
           
        # backward pass, single & joint marginal calculation, sufficient stats
        for i in np.linspace(n_samples-1,0,n_samples):
            
            beta_after     = np.dot(pr_trans,beta_before*pr_x[i,:])
            
            # marginal distribution of latent variable, given observed variables
            marginal       = alpha[i,:]*beta_before                         
            
            if i > 0:
                # joint marginal of two latent variables, given observed ones
                joint_marginal = pr_trans * np.outer(alpha[i-1,:], pr_x[i,:]*beta_before)
            
                # iterative update of posterior for transitional probability
                trans += joint_marginal
            else:
                # update for posterior of intial latent variable
                start += marginal
            
            # iterative update of sufficient statistics for emission probs
            suff_stats     = self._suff_stats_update(suff_stats,X[i,:],marginal)
            beta_before    = _normalise(beta_after)
        
        return trans, start, suff_stats
         
   
        
    def _forward_single_chain(self, pr_start, pr_trans, pr_x):
        '''
        Performs forward pass ('filter') on single Hidden Markov Model chain
        '''
        n_samples = pr_x.shape[0]
        alpha = np.zeros(pr_x.shape)
        
        # forward pass
        alpha[0,:] = _normalise(pr_x[0,:] * pr_start)
        for i in range(1,n_samples):
            alpha[i,:] = _normalise( np.dot(pr_trans.T,alpha[i-1,:]) * pr_x[i,:])
        return alpha
        
        
        
        
class VBBernoulliHMM(VBHMM):
    '''
    Bayesian Hidden Markov Models with Bernoulli Emission probabilities
    
    Parameters
    ----------
    n_hidden: int, optional (DEFAULT = 2)
       Number of hidden states
       
    n_iter: int, optional (DEFAULT = 100)
       Number of iterations of VBEM algorithm
       
    tol: float, optional (DEFAULT = 1e-3)
       Convergence threshold
       
    alpha_start: float, optional (DEFAULT = 1.0)
       Concentration parameter for distibution of starting point of HMM
       
    alpha_trans: float, optional (DEFAULT = 1.0)
       Concentration parmater for transition probability matrix parameters
              
    verbose: bool, optional (DEFAULT = False)
       If True prints intermediate results and progress report at each iteration
    ''' 
    def __init__(self, n_hidden = 2, n_iter = 100, init_params = {}, tol = 1e-3,
                 alpha_start = 1, alpha_trans = 5 , alpha_succes = 1, alpha_fail = 1,
                 verbose = False):
        super(VBBernoulliHMM,self).__init__(n_hidden, n_iter, init_params, tol,
                                            alpha_start, alpha_trans, verbose)
        self.alpha_succes = alpha_succes
        self.alpha_fail   = alpha_fail
         
         
    
    def _init_params(self,*args):
        ''' 
        Initialise parameters of Bayesian Bernoulli HMM
        '''
        n_features         = args[0]
        pr_start, pr_trans = super(VBBernoulliHMM,self)._init_params()
        pr_succes = np.random.random([self.n_hidden, n_features])* self.alpha_succes
        pr_fail   = np.random.random([self.n_hidden, n_features])* self.alpha_fail
        return pr_start, pr_trans , {'success_prob': pr_succes, 'fail_prob': pr_fail}
        
    
    
    def _emission_probs_params(self, emission_params, X):
        '''
        Compute emission probabilities
        '''
        success = emission_params['success_prob']
        fail    = emission_params['fail_prob']
        log_total   = psi(success + fail)
        log_success = psi(success) -  log_total
        log_fail    = psi(fail)    -  log_total
        log_normaliser = np.logaddexp(log_success, log_fail)
        log_pr_succes = log_success - log_normaliser
        log_pr_fail   = log_fail    - log_normaliser
        return np.exp(safe_sparse_dot(X,log_pr_succes) + 
                      safe_sparse_dot(np.ones(X.shape) - X, log_pr_fail))
        
        
                                    
    def _vbm_emission_params(self,emission_params_prior, emission_params, sf_stats):
        '''
        Peerforms vbm step for parameters of emission probabilities
        '''
        emission_params['success_prob'] = emission_params_prior['success_prob'] + sf_stats[0]
        fail_delta = (sf_stats[1]  - sf_stats[0].T).T
        emission_params['fail_prob'] = emission_params_prior['fail_prob'] + fail_delta
        return emission_params
                      
        
    
    def fit(self,X,chain_index):
        '''
        Fits Bayesian Hidden Markov Model
        
        Parameters
        ----------
        X: array-like or csr_matrix of size (n_samples, n_features)
           Data Matrix
           
        Returns
        -------
        object: self
          self
        '''
        trans_params, start_params, emission_params = super(VBBernoulliHMM,self)._fit(X)
        
        
        
    def filter(self,X):
        '''
        Performs filtering on matrix of observations
        
        Parameters
        ----------
        X: array-like or csr_matrix of size (n_samples, n_features)
           Data Matrix

        '''
        pass
        
        
        
    def _init_suff_stats(self,n_features):
        ''' 
        Initialise sufficient statistics for Bayesian Bernoulli HMM
        '''
        return [ np.zeros( [self.n_hidden, n_features] ), 
                 np.zeros( self.n_hidden ) ]
                 
    
    def _suff_stats_update(self,sf_stats, x, marginal):
        '''
        Updates sufficient statistics within backward pass in HMM
        '''
        sf_stats[0] += np.outer(marginal,x)
        sf_stats[1] += marginal
        return sf_stats
        
    
    def _check_convergence(self):
        return False


        
if __name__ == "__main__":
    X = np.array([[0,0],[0,0],[0,0],[1,1],[1,1],[1,1],[0,0],[0,0],[0,0],[1,1],
                  [1,1],[1,1],[0,0],[0,0],[0,0],[1,1],[1,1]])
    X1 = np.array([[0,0],[0,0],[0,0],[1,1],[1,1],[1,1],[0,0],[0,0],[0,0],[1,1],
                  [1,1],[1,1],[0,0],[0,0],[0,0],[1,1],[1,1]])
                  
    bhmm = VBBernoulliHMM(n_iter = 200)
    start_params, trans_params, emission_params = bhmm._fit(X)
    pr_start, pr_trans, pr_x = bhmm._probs_params(start_params,trans_params,emission_params,X)
    alpha = bhmm._forward_single_chain(pr_start, pr_trans, pr_x)
           

    
        
    
    
        
