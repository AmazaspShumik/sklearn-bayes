from scipy.special import psi,gammaln
from scipy.misc import logsumexp
from scipy.linalg import pinvh
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.mixture import VBGMM
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np



def _normalise(M):
    ''' Make matrix or vector stochastic (i.e. normalise by row to 1)'''
    if len(M.shape) == 1:
        return M / np.sum(M)
    return M / np.sum(M, axis = 1)
    
    
    
def _get_chain(X,index = []):
    ''' Generates separate chains'''
    from_idx = 0
    if len(index)==0:
        yield X[from_idx:,:]
    else:
        for idx in index:
            yield X[from_idx:idx,:]
            from_idx = idx
        if from_idx != X.shape[0]-1:
            yield X[from_idx:(X.shape[0]-1),:]
            
     
                   
def _check_shape_sign(x,shape,shape_message, sign_message):
    ''' Checks shape and sign of input, raises error'''
    if x.shape != shape:
        raise ValueError(shape_message)
    if np.sum( x < 0 ) > 0:
        raise ValueError(sign_message)
    
                  

class VBHMM(BaseEstimator):
    '''
    Superclass for Variational Bayesian Hidden Markov Models.
    
    This class implements methods that do not depend on emission probabilities,
    all inference steps that explicitly use emission pdf are implemented
    in subclasses.
    '''
    
    def __init__(self, n_hidden = 2, n_iter = 100, init_params = None, tol = 1e-3,
                 alpha_start = 10, alpha_trans = 10, verbose = False):
        self.n_hidden    = n_hidden
        self.n_iter      = n_iter
        self.init_params = init_params
        self.tol         = tol
        self.verbose     = verbose
        self.alpha_start = alpha_start
        self.alpha_trans = alpha_trans
        
    
    def _init_params(self):
        ''' 
        Reads user defined parameters and checks their validity. In case parameters 
        are not defined they are randomly initialised
        '''
        # initial distribution
        sign_message = ('Parameters of Dirichlet Distribution can not be negative')
        if 'start' in self.init_params:
            pr_start = self.init_params['initial']
            
            # check shape and nonnegativity
            shape = (self.n_hidden,)
            shape_message = ('Parameters of distribution of initial state should have shape'
                             ' {0}, oberved shape is {1}').format(shape,pr_start.shape[0])
            _check_shape_sign(pr_start,shape, shape_message, sign_message)
        else:
            pr_start = np.random.random(self.n_hidden) * self.alpha_start
            
        # matrix of transition probabilities
        if 'transition' in self.init_params:
            pr_trans = self.init_params['transition']

            # check shape and nonnegativity
            shape = (self.n_hidden,self.n_hidden)
            shape_message = ('Parameters for transition probability distribution should have '
                             'shape {0}, observed shape is {1}').format(shape,pr_trans.shape)
            _check_shape_sign(pr_start,shape, shape_message, sign)
        else:
            pr_trans = np.random.random( [self.n_hidden, self.n_hidden] ) * self.alpha_trans
        return pr_start, pr_trans


    
    def _log_probs_params(self, start_params, trans_params, emission_params, X):
        '''
        Compute log probabilities : emission, initial, transition using parameters
        '''
        log_pr_start = psi(start_params) - psi(np.sum(start_params))
        log_pr_start-= logsumexp(log_pr_start)
        log_pr_trans = psi(trans_params) - psi(np.sum(trans_params,1))
        log_pr_trans-= logsumexp(log_pr_trans,1,keepdims = True)
        log_pr_x     = self._emission_log_probs_params(emission_params,X)
        return log_pr_start, log_pr_trans, log_pr_x
       


    def _probs_params(self, start_params, trans_params, emission_params, X):
        '''
        Compute probabilities: emission, initial, transition using parameters
        '''
        log_pr_start, log_pr_trans, log_pr_x = self._log_probs_params( start_params, 
                                                      trans_params, emission_params, X)
        print np.argmax(log_pr_x,1)
        return np.exp(log_pr_start), np.exp(log_pr_trans),  log_pr_trans, log_pr_x
    
        
                    
    def _fit(self, X, chain_indices = []):
        '''
        Fits Hidden Markov Model with unspecified emission probability
        '''
        n_samples, n_features = X.shape

        # initialise parameters (log-scale!!!)
        start_params, trans_params, emission_params = self._init_params(n_features,X)
        trans_params_prior = np.copy(trans_params)
        start_params_prior = np.copy(start_params)
        emission_params_prior = deepcopy(emission_params)
        
        for i in range(self.n_iter):
            
            # statistics that accumulate data for m-step
            sf_stats = self._init_suff_stats(n_features)
            trans = np.zeros([self.n_hidden, self.n_hidden])
            start = np.zeros(self.n_hidden)

            # probabilies for initialised parameters           
            log_pr_start, pr_trans, log_pr_trans, log_pr_x = self._probs_params(start_params, 
                                                                  trans_params,
                                                                  emission_params, X)
        
            for zx in _get_chain(X,chain_indices):
                
                alpha, scaler = self._forward_single_chain( log_pr_start, pr_trans, log_pr_x)
                trans, start, sf_stats = self._vbe_step_single_chain(zx,alpha,scaler,pr_trans,
                                                          log_pr_trans,log_pr_x,sf_stats, 
                                                          trans, start)
                
            # log parameters of posterior distributions of parameters
            trans_params, start_params, emission_params = self._vbm_step(trans,start,
                                                          sf_stats, emission_params,
                                                          trans_params_prior,
                                                          emission_params_prior,
                                                          start_params_prior) 
            if self._check_convergence():
                break
                
        self._start_params_    = start_params
        self._trans_params_    = trans_params
        self._emission_params_ = emission_params
                
                
        
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
        
        

    def _vbe_step_single_chain(self, X, alpha, scaler, pr_trans, log_pr_trans, log_pr_x, suff_stats, trans, start):
        '''
        Performs backward pass, at the same time computes marginal & joint marginal
        and updates sufficient statistics for VBM step
        '''
        beta_before   = np.ones(self.n_hidden)
        n_samples     = X.shape[0]
           
        # backward pass, single & joint marginal calculation, sufficient stats
        for i in np.linspace(n_samples-1,0,n_samples):
            
            # recursively compute beta (start from the end of sequence, where beta = 1)
            beta_after     = np.log(np.dot(pr_trans,beta_before)) + log_pr_x[i,:]
            
            # marginal distribution of latent variable, given observed variables
            #marginal       = _normalise(alpha[i,:]*beta_before)  
            marginal = alpha[i,:] *beta_before                    
            
            if i > 0:
                # joint marginal of two latent variables, given observed ones
                delta = np.log(np.outer(alpha[i-1,:], beta_before)) + log_pr_x[i,:]
                joint_marginal = log_pr_trans + delta
                joint_marginal = np.exp(joint_marginal) # - logsumexp(joint_marginal))                

                # iterative update of posterior for transitional probability
                trans += joint_marginal
            else:
                # update for posterior of intial latent variable
                start += marginal
            
            #print "ALPHA & BETA"
            #print alpha[i,:]
            #print beta_after
            #print "Marginals"
            #print marginal
            
            # iterative update of sufficient statistics for emission probs
            suff_stats     = self._suff_stats_update(suff_stats,X[i,:],marginal)
            beta_before    = np.exp(beta_after - scaler[i])
        
        return trans, start, suff_stats
        
          
        
    def _forward_single_chain(self, log_pr_start, pr_trans, log_pr_x):
        '''
        Performs forward pass ('filter') on single Hidden Markov Model chain
        '''
        n_samples = log_pr_x.shape[0] 
        alpha     = np.zeros(log_pr_x.shape)
        alpha[0,:] = log_pr_x[0,:] + log_pr_start
        alpha[0,:] -= logsumexp(alpha[0,:])
        alpha[0,:] = np.exp(alpha[0,:])
        scaler     = np.zeros(n_samples)
        for i in range(1,n_samples):
            alpha[i,:] = np.dot(pr_trans.T,alpha[i-1,:]) 
            alpha[i,:] = np.log(alpha[i,:]) + log_pr_x[i,:] 
            scaler[i]  = logsumexp(alpha[i,:])
            alpha[i,:] = np.exp(alpha[i,:] - scaler[i])
        return alpha, scaler
        
        
        
    def _viterbi(self, log_pr_x, log_pr_trans, log_pr_start, X):
        '''
        Computes most probable sequence of states using viterbi algorithm
        '''
        n_samples     = log_pr_x.shape[0]
        best_states   = np.zeros(n_samples)
        max_prob      = np.zeros([n_samples,self.n_hidden])
        argmax_state  = np.zeros([n_samples,self.n_hidden])
        max_prob[0,:] = log_pr_x[0,:] + log_pr_start
        
        # forward pass of viterbi algorithm
        for t in xrange(1,n_samples):
            
            # precompute some values
            delta = max_prob[t-1,:] + log_pr_trans
            
            # compute log probs (not normalised) for sequence of states
            max_prob[t,:] = log_pr_x[t,:] + np.max(delta,1)
            
            # most likely previous state on the most probable path
            argmax_state[t,:] = np.argmax((log_pr_x[t,:] + delta.T).T,0)
            
        # backtrack
        best_states[n_samples-1] = np.argmax(max_prob[n_samples-1,:])
        for j in xrange(1,n_samples):
            t = n_samples - j - 1
            best_states[t] = argmax_state[t,best_states[t+1]]

        print 'argmax state'
        print argmax_state
        print max_prob
        return best_states

        
        
    def predict_proba(self,X):
        '''
        Performs filtering on matrix of observations
        
        Parameters
        ----------
        X: array-like or csr_matrix of size (n_samples, n_features)
           Data Matrix
           
        Returns
        -------
        alpha: numpy array of size (n_samples, n_hidden)
           Belief states for each observation in X matrix
        '''
        check_is_fitted(self,'_start_params_')
        log_pr_start,pr_trans,log_pr_trans,log_pr_x = self._probs_params(self._start_params_, 
                                                      self._trans_params_, 
                                                      self._emission_params_, X)
        alpha = self._forward_single_chain( log_pr_start, pr_trans, log_pr_x)
        return alpha
        
        
        
    def predict(self,X):
        '''
        Predicts cluster for test data
        
        Parameters
        ----------
        X: array-like or csr_matrix of size (n_samples, n_features)
           Data Matrix
           
        Returns
        -------
        : numpy array of size (n_samples,)
           Hidden state index
        '''
        check_is_fitted(self,'_start_params_')
        log_pr_start, log_pr_trans, log_pr_x = self._log_probs_params(self._start_params_,
                                                                      self._trans_params_,
                                                                      self._emission_params_,X)
        return self._viterbi(log_pr_x, log_pr_trans, log_pr_start, X)
        
               
        
        
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
       
    init_params: dictionary, optional (DEFAULT = {} )
       
       'start': numpy array of size (n_hidden,)
             Parameters of prior of initial state distribution
        
       'transition': numpy array of size (n_hidden,n_hidden)
             Parameters of prior of transition matrix distribution
             
       'alpha': numpy array of size (n_hidden, n_features)
             First shape parameter for beta distribution (prior of success probs)
             
       'beta': numpy array of size (n_hidden, n_features)
             Second shape parameter for beta distribution (prior of success probs)
                    
    alpha_start: float, optional (DEFAULT = 1.0)
       Concentration parameter for distibution of starting point of HMM
       
    alpha_trans: float, optional (DEFAULT = 1.0)
       Concentration parmater for transition probability matrix parameters
              
    verbose: bool, optional (DEFAULT = False)
       If True prints intermediate results and progress report at each iteration
       
       
    Attributes
    ----------
    
    ''' 
    def __init__(self, n_hidden = 2, n_iter = 100, init_params = {}, tol = 1e-3,
                 alpha_start = 10, alpha_trans = 10 , alpha_succes = 5, alpha_fail = 5,
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
        
        # check user defined parameters for prior, if not provided generate your own
        shape         = (self.n_hidden,n_features)
        shape_message = ('Parameters for prior of success probabilities should have shape '
                         '{0}').format(shape)
        sign_message  = 'Parameters of bets distriboution can not be negative'
        
        # parameter for success probs
        if 'alpha' in self.init_params:
            pr_success = self.init_params['alpha']
            _check_shape_sign(pr_success,shape, shape_message, sign_message)            
        else:
            pr_succes = np.random.random([self.n_hidden, n_features])* self.alpha_succes
            
        # parameters for fail probs
        if 'beta' in self.init_params:
            pr_fail = self.init_params['beta']
            _check_shape_sign(pr_fail,shape, shape_message, sign_message)
        else:
            pr_fail   = np.random.random([self.n_hidden, n_features])* self.alpha_fail
            
        return pr_start, pr_trans , {'success_prob': pr_succes, 'fail_prob': pr_fail}    
    
    
    
    def _emission_log_probs_params(self, emission_params, X):
        '''
        Computes log of emission probabilities
        '''
        success = emission_params['success_prob']
        fail    = emission_params['fail_prob']
        log_total   = psi(success + fail)
        log_success = psi(success) -  log_total
        log_fail    = psi(fail)    -  log_total
        
        # normalisation is not neccesary here
        #log_normaliser = np.logaddexp(log_success, log_fail)
        #log_pr_success = log_success - log_normaliser
        #log_pr_fail   = log_fail    - log_normaliser
        #return safe_sparse_dot(X,log_pr_success.T) + safe_sparse_dot(np.ones(X.shape) - X, log_pr_fail.T)
        return safe_sparse_dot(X,log_success.T) + safe_sparse_dot(np.ones(X.shape) - X, log_fail.T)
                                  
    def _vbm_emission_params(self,emission_params_prior, emission_params, sf_stats):
        '''
        Performs vbm step for parameters of emission probabilities
        '''
        emission_params['success_prob'] = emission_params_prior['success_prob'] + sf_stats[0]
        fail_delta = (sf_stats[1]  - sf_stats[0].T).T
        emission_params['fail_prob'] = emission_params_prior['fail_prob'] + fail_delta
        return emission_params

        
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
                      
    
    def fit(self,X,chain_index = []):
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
        super(VBBernoulliHMM,self)._fit(X, chain_index)
        return self



    def _check_convergence(self):
        return False
        

        

class VBGaussianHMM(VBHMM):
    '''
    Variational Bayesian Hidden Markov Model with Gaussian emission probabilities
    
    Parameters
    ----------
    n_hidden: int, optional (DEFAULT = 2)
       Number of hidden states
       
    n_iter: int, optional (DEFAULT = 100)
       Number of iterations of VBEM algorithm
       
    tol: float, optional (DEFAULT = 1e-3)
       Convergence threshold
       
    alpha_start: float, optional (DEFAULT = 2.0)
       Concentration parameter for distibution of starting point of HMM
       
    alpha_trans: float, optional (DEFAULT = 2.0)
       Concentration parmater for transition probability matrix parameters
              
    verbose: bool, optional (DEFAULT = False)
       If True prints intermediate results and progress report at each iteration
    '''
    def __init__(self, n_hidden = 2, n_iter = 100, init_params = {}, tol = 1e-3,
                 alpha_start = 2, alpha_trans = 2 , verbose = False):
        super(VBGaussianHMM,self).__init__(n_hidden, n_iter, init_params, tol,
                                           alpha_start, alpha_trans, verbose)
     
    
    
    def _init_params(self,*args):
        ''' 
        Initialise parameters of Bayesian Gaussian HMM
        '''
        d,X         = args
        pr_start, pr_trans = super(VBGaussianHMM,self)._init_params()

        # initialise prior on means & precision matrices
        if 'means' in self.init_params:
            means0   = self.init_params['means']
        else:
            kms = KMeans(n_init = 2, n_clusters = self.n_hidden)
            means0 = kms.fit(X).cluster_centers_
            
        if 'covar' in self.init_params:
            scale_inv0 = self.init_params['covar']
            scale0     = pinvh(scale_inv0)
        else:
            # heuristics to define broad prior over precision matrix
            diag_els   = np.abs(np.max(X,0) - np.min(X,0))
            scale_inv0 = np.diag( diag_els  )
            scale0     = np.diag( 1./ diag_els )

        if 'dof' in self.init_params:
            dof0 = self.init_params['dof']
        else:
            dof0 = d
            
        if 'beta' in self.init_params:
            beta0 = self.init_params['beta']
        else:
            beta0 = 1e-3
        
        # checks initialisation errors in case parameters are user defined
        assert dof0 >= d,( 'Degrees of freedom should be larger than '
                                'dimensionality of data')
        assert means0.shape[0] == self.n_hidden,('Number of centrods defined should '
                                                     'be equal to number of components')
        assert means0.shape[1] == d,('Dimensioanlity of means and data '
                                          'should be the same')

        scale   = np.array([np.copy(scale0) for _ in range(self.n_hidden)])
        dof     = dof0*np.ones(self.n_hidden)
        beta    = beta0*np.ones(self.n_hidden)
        
        # if user did not define initialisation parameters use KMeans
        return pr_start, pr_trans, {'means':means0,'scale':scale,'beta': beta,
                                    'dof':dof,'scale_inv0':scale_inv0}
        
        
        
    def _init_suff_stats(self,n_features):
        ''' 
        Initialise sufficient statistics for Bayesian Bernoulli HMM
        (Similar to 10.51 in Bishop(2006) , but instead of weighted avergae we 
        use weighted sum to avoid underflow issue)
        '''
        return [ 
                 np.zeros(self.n_hidden),
                 np.zeros( [self.n_hidden, n_features] ),
                 np.zeros( [self.n_hidden, n_features, n_features] )
               ]
                 
              
                    
    def _suff_stats_update(self,sf_stats, x, marginal):
        '''
        Updates sufficient statistics within backward pass in HMM
        '''
        xx  = np.outer(x,x)
        sf_stats[2]  = [XX + marginal[k]*xx for k,XX in enumerate(sf_stats[2])]
        sf_stats[1] += np.outer(marginal,x)
        sf_stats[0] += marginal
        return sf_stats
       
         
         
    def _vbm_emission_params(self,emission_params_prior, emission_params, sf_stats):
        '''
        Performs vbm step for parameters of emission probabilities
        '''
        Nk,Xk,Sk = sf_stats
        beta0, means0 = emission_params_prior['beta'], emission_params_prior['means']
        emission_params['beta']  =  beta0 + Nk
        emission_params['means'] = ((beta0*means0.T + Xk.T ) / emission_params['beta']).T
        emission_params['dof']   = emission_params_prior['dof'] + Nk + 1
        scale_inv0               = emission_params_prior['scale_inv0']
        for k in range(self.n_hidden):
            emission_params['scale'][k] = pinvh( scale_inv0 + (beta0*Sk[k] + Nk[k]*Sk[k] - 
                                       np.outer(Xk[k],Xk[k]) - 
                                       beta0*np.outer(means0[k] - Xk[k],means0[k])) /
                                       (beta0 + Nk[k]) )
            print 'after M-step'
            print emission_params['scale']
            
        return emission_params        
            
        
        
    def _emission_log_probs_params(self, emission_params, X):
        '''
        Computes log of Gaussian emission probs for approximating distribution
        '''
        # retrieve releavant parameters from emission_params
        m,d   = X.shape
        scale = emission_params['scale']
        dof   = emission_params['dof']
        means = emission_params['means']
        beta  = emission_params['beta']
        log_probs = np.zeros([m,self.n_hidden])
         
        for k in range(self.n_hidden):
            
            # calculate expectation of logdet of precision matrix
            print 'Scaling matrix'
            print scale[k]
            scale_logdet   = np.linalg.slogdet(scale[k])[1]
            e_logdet_prec  = sum([psi(0.5*(dof[k]+1-i)) for i in range(1,d+1)])
            e_logdet_prec += scale_logdet + d*np.log(2)
           
            # calculate expectation of quadratic form (x-mean_k)'*precision_k*(x - mean_k)
            x_diff         = X - means[k,:]
            e_quad_form    = np.sum( np.dot(x_diff,scale[k,:,:])*x_diff, axis = 1 )
            e_quad_form   *= dof[k]
            e_quad_form   += d / beta[k] 
            log_probs[:,k] = 0.5*(e_logdet_prec - e_quad_form)
        
        return log_probs - logsumexp(log_probs)
        
        
    def _check_convergence(self):
        return False
        
        
        
    def fit(self,X,chain_index = []):
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
        super(VBGaussianHMM,self)._fit(X, chain_index)
        return self
        
        

if __name__ == "__main__":
    
    ## imports used in testing
    #import matplotlib.pyplot as plt
    #
    # testing Bernoulli HMM
    #X = np.array([[0,0,0],[0,0,0],[0,0,0],[1,1,1],[1,1,1],[1,1,1],[0,0,0],[0,0,0],
    #              [0,0,0],[1,1,1],[1,1,1],[1,1,1],[0,0,0],[0,0,0],[0,0,0],[1,1,1],
    #              [1,1,1]])
    #X1 = np.array([[0,0],[0,0],[0,0],[1,1],[1,1],[1,1],[0,0],[0,0],[0,0],[1,1],
    #              [1,1],[1,1],[0,0],[0,0],[0,0],[1,1],[1,1]])
    #              
    #bhmm = VBBernoulliHMM(n_iter = 20)
    #bhmm.fit(X)
    ##start_params, trans_params, emission_params = bhmm._init_params(3,X)
    #
    ## test filtering 
    #alpha = bhmm.predict(X)
    #probs = bhmm.predict_proba(X)
    #
    #
    

    ## test viterbi    
    #log_pr_start, log_pr_trans, log_pr_x = bhmm._log_probs_params(bhmm._start_params_,bhmm._trans_params_,
    #                                              bhmm._emission_params_,X)
    #
    #best_states = bhmm.predict(X)
    #print best_states
    #
    #testing Gaussian HMM
    X = np.random.random([200,2])
    X[0:100,:] += 2
    plt.plot(X[:,0],X[:,1],'r+')
    plt.show()
    
    ghmm = VBGaussianHMM(n_iter = 10)
    ghmm.fit(X)
    alpha = ghmm.predict(X)
    
    
    
    
           

    
        
    
    
        
