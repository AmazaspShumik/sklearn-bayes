
from sklearn.base import BaseEstimator
import numpy as np

def _normalise(M):
    ''' Make matrix or vector stochastic (i.e. normalise by row to 1)'''
    if len(M.shape) == 1:
        return M / np.sum(M)
    return M / np.sum(M, axis = 1)
    

class VBHMM(BaseEstimator):
    '''
    Superclass for implementation of Variational Bayesian Hidden Markov 
    Models.
    
    This class implements inference steps that does not depend on form 
    of emission probabilities.
    '''
    
    def __init__(self, n_hidden = 2, n_iter = 100, init_params = None, tol = 1e-3,
                 verbose = False):
        self.n_hidden    = n_hidden
        self.n_iter      = n_iter
        self.init_params = init_params
        self.tol         = tol
        self.verbose     = verbose
        
    
    def _init_params(self):
        ''' Initialise parameters'''
        pass
    
    
    def _fit(self, X, chain_indices):
        '''
        Fits Hidden Markov Model
        '''
        for i in range(self.n_iter):
            
            for X in X_chains:
                alpha = self._forward_single_chain( pr_start, pr_trans, pr_x, alpha)
                trans, start, sf_stats = self._vbe_step_single_chain(X,alpha,pr_trans,
                                                                     suff_stats)
                        self._vbm_step_update()
                                                                     
                
            
        
        
    def _vbm_step_update(self, pr_trans_prior, pr_start_prior, pr_trans_post,
                               pr_start_post):
        '''
        Computes approximating distribution for posterior of parameters
        '''
        pass
        
        
        

    def _vbe_step_single_chain(self, X, alpha, pr_trans, pr_x, suff_stats):
        '''
        Performs backward pass, at the same time computes marginal & joint marginal
        and updates sufficient statistics for VBM step
        '''
        beta_before   = np.ones(self.n_hidden_states)
        n_samples     = X.shape[0]
        pr_trans_post = np.zeros(self.n_hidden, self.n_hidden)
        pr_start_post = np.zeros(self.n_hidden)
           
        # backward pass, single & joint marginal calculation, sufficient stats
        for i in np.linspace(n_samples-1,0,n_samples):
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
            suff_stats     = self._update_suff_stats(suff_stats,X,marginal)
            beta_before    = beta_after
        
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
        
