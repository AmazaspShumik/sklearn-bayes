
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
    
    def __init__(self):
        pass
    
    
    def _vbe_step(self):
        pass
        
        
    def _vbm_step_single_chain(self, alpha_A, alpha_pr, e_pr, e_A):
        '''
        Computes approximating distribution for posterior of parameters
        
        Parameters
        ----------
        A: numpy array of size [k,k] (where k is number of states)
           Transition Matrix
           
        pr: numpy array of size (k,)
           Probability distribution of initial state in HMM
           
        phi: list 
           List of parameters specific to emission probability distribution
        '''
        resps
        
        
        
        
    def _vbe_step_posterior_n(self,alpha, beta, pr_x, n):
        '''
        Calculates conditional distributions of latent variables required
        for 
        '''
        resps = alpha*beta
        
        # use generator for approximating posterior 
        def 
        
        
     
        
        
        
    def _forward_backward_single_chain(self, pr_start, pr_trans, pr_x, alpha, beta):
        '''
        Performs forward-backward pass on single Hidden Markov Model chain
        
        Parameters
        ----------
        pr_start: numpy array of size (n_hidden_states,)
            Hidden state probabilities at start of HMM
            
        pr_trans: numpy array of size (n_hidden_states,n_hidden_states)
           Transition matrix for hidden states
           
        pr_x: numpy array of size (n_samples,n_hidden_states)
            Conditional probability of observed variable given hidden state
            
        Returns
        -------
        alpha: numpy array of size (n_samples,)
           p(z_t | X_(1:t))
           
        beta: numpy array of size (n_sampes,)
           p(X_(t+1:T)| z_t) / p(X_(t+1:T)| X_(1:t))
        '''
        n_samples = len(alpha)
        
        # forward pass
        alpha[0,:] = _normalise(pr_x[0,:] * pr_start)
        for i in range(1,n_samples):
            alpha[i,:] = _normalise( np.dot(pr_trans.T,alpha[i-1,:]) * pr_x[i,:] )
            
        # backward pass
        for j in range(1,n_samples):
            i       = n_samples - j 
            beta[i] = np.dot(pr_trans,beta[i+1,:]*pr_x[i+1,:])

        return alpha,beta
        
        
        
        
        

        
        
    def _e_step_generic(self):
        pass
        