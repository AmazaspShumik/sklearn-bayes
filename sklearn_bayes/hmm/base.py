from scipy.special import psi,gammaln
from scipy.misc import logsumexp
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
import numpy as np



def _normalise(M):
    ''' Make matrix or vector stochastic (i.e. normalise by row to 1)'''
    if len(M.shape) == 1:
        return M / np.sum(M)
    return M / np.sum(M, axis = 1)
    
    
def _get_chain(X,index):
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
        log_pr_trans-= logsumexp(log_pr_trans)
        log_pr_x     = self._emission_probs_params(emission_params,X)
        return np.exp(log_pr_start), np.exp(log_pr_trans), np.exp(log_pr_x)
        
                    
    def _fit(self, X, chain_indices):
        '''
        Fits Hidden Markov Model
        '''
        n_samples = X.shape[0]
        alpha = np.zeros(n_samples, self.n_hidden)       

        # initialise parameters (log-scale!!!)
        trans_params, start_params, emission_params = self._init_params()
        
        for i in range(self.n_iter):
            pr_trans_post = np.copy(pr_trans_prior)
            pr_start_post = np.copy(pr_start_prior)
            sf_stats_post = self._init_suff_stats()
            
            # probabilies for initialised parameters
            pr_start, pr_trans, pr_x = self._probs_params(start_params, trans_params,
                                                          emission_params, X)
            for X in _get_chain(X,chain_indices):
                
                alpha = self._forward_single_chain( pr_start, pr_trans, pr_x, alpha)
                trans, start, sf_stats = self._vbe_step_single_chain(X,alpha,pr_trans,
                                                                     suff_stats)
                pr_trans_post += trans
                pr_start_post += start
                sf_stats_post  = self._suff_stats_update_new_chain(sf_stats,X,)
                
            # log parameters of posterior distributions of parameters
            trans_params, start_params, emission_params = self._vbm_step(pr_trans_post, 
                                                           pr_start_post, sf_stats_post)
            
            if self._check_convergence():
                break
                
        
    def _vbm_step(self, pr_trans_post, pr_start_post, sf_stats_post):
        '''
        Computes approximating distribution for posterior of parameters
        '''
        pass 
        

    def _vbe_step_single_chain(self, X, alpha, pr_trans, pr_x, suff_stats):
        '''
        Performs backward pass, at the same time computes marginal & joint marginal
        and updates sufficient statistics for VBM step
        '''
        beta_before   = np.ones(self.n_hidden) / self.n_hidden
        n_samples     = X.shape[0]
        pr_trans_post = np.zeros(self.n_hidden, self.n_hidden)
        pr_start_post = np.zeros(self.n_hidden)
           
        # backward pass, single & joint marginal calculation, sufficient stats
        for i in np.linspace(n_samples-1,0,n_samples):
            
            # ???? normalise
            beta_after     = np.dot(pr_trans,beta_before*pr_x[i,:])
            
            # marginal distribution of latent variable, given observed variables
            marginal       = alpha[i,:]*beta_before                         
            
            if i > 0:
                # joint marginal of two latent variables, given observed ones
                joint_marginal = pr_trans * np.outer(alpha[i-1,:], pr_x[i,:]*beta_before)
            
                # iterative update of posterior for transitional probability
                pr_trans_post += joint_marginal
            else:
                # update for posterior of intial latent variable
                pr_start_post += marginal
                
            
            # iterative update of sufficient statistics for emission probs
            suff_stats     = self._suff_stats_update(suff_stats,X,marginal)
            beta_before    = _normalise(beta_after)
        
        return pr_trans_post, pr_start_post, suff_stats
         
   
        
    def _forward_single_chain(self, pr_start, pr_trans, pr_x, alpha):
        '''
        Performs forward pass ('filter') on single Hidden Markov Model chain
        '''
        n_samples = len(alpha)
        
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
    n_hidden: int , optional (DEFAULT = 2)
       Number of hidden states
       
    n_iter: 
    '''
    
    def __init__(self, n_hidden = 2, n_iter = 100, init_params = {}, tol = 1e-3,
                 alpha_start = 1, alpha_trans = 1 , alpha_succes = 1, verbose = False):
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
        return pr_start, pr_succes, {'success_prob': pr_succes, 'fail_prob': pr_fail}
        
    
    def _emission_probs_params(self, emission_params, X):
        '''
        C
        '''
        success = emission_params['success_prob']
        fail    = emission_params['fail_prob']
        log_total = psi(success + fail)
        log_pr_success = psi(success) -  log_total
        log_pr_fail    = psi(fail)    -  log_total
        pr_succes = np.exp(log_pr_success - np.logaddexp(log_pr_success, log_pr_fail))
        return safe_sparse_dot(X,pr_succes)
        
        
    
    def _fit(self,X):
        '''
        Fits Bayesian Hidden Markov Model
        '''
        super(VBBernoulliHMM,self)._fit(X)
        
        
        
    def _init_suff_stats(self,n_samples):
        ''' 
        Initialise sufficient statistics for Bayesian Bernoulli HMM
        '''
        return [ np.zeros( [self.n_hidden, n_samples] ), 
                 np.zeros( self.n_hidden ) ]
                 
    
    def _suff_stats_update(self):
        '''
        Updates sufficient statistics within backward pass in HMM
        '''
                 
    
    def _suff_stats_update_new_chain(self):
        '''
        Updates sufficient statistics after observing new HMM
        '''
        
        
if __name__ == "__main__":
    bhmm = VBBernoulliHMM()

    
        
    
    
        
