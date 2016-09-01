import numpy as np
from scipy.sparse import isspmatrix
from sklearn.utils.validation import check_is_fitted
from sklearn.cluster import KMeans
from sklearn.utils import check_array
from scipy.linalg import pinvh




def _check_shape_sign(x,shape,shape_message, sign_message):
    ''' Checks shape and sign of input, raises error'''
    if x.shape != shape:
        raise ValueError(shape_message)
    if np.sum( x < 0 ) > 0:
        raise ValueError(sign_message)
        

def _get_classes(X):
    '''Finds number of unique elements in matrix'''
    if isspmatrix(X):
        v = X.data
        if len(v) < X.shape[0]*X.shape[1]:
            v = np.hstack((v,np.zeros(1)))
        V     = np.unique(v)
    else:
        V     = np.unique(X)
    return V      
  
        
class BernoulliMixture(object):
          
          
    def _init_params(self, X):
        ''' 
        Initialise parameters of Bernoulli Mixture Model
        '''
        # check user defined parameters for prior, if not provided generate your own
        shape         = (X.shape[1], self.n_components)
        shape_message = ('Parameters for prior of success probabilities should have shape '
                         '{0}').format(shape)
        sign_message  = 'Parameters of beta distribution can not be negative'
        
        # parameter for success probs
        if 'a' in self.init_params:
            pr_success = self.init_params['a']
            _check_shape_sign(pr_success,shape,shape_message,sign_message)            
        else:
            pr_success = np.random.random([X.shape[1],self.n_components]) * self.a
            
        # parameters for fail probs
        if 'b' in self.init_params:
            pr_fail = self.init_params['b']
            _check_shape_sign(pr_fail,shape,shape_message,sign_message)
        else:
            pr_fail   = np.random.random([X.shape[1],self.n_components]) * self.b
        c,d = pr_success, pr_fail
        c_init, d_init = np.copy(c), np.copy(d)
        return {'c':c,'d':d,'c_init':c_init,'d_init':d_init}
        
        
        
    def _check_X(self,X):
        ''' 
        Checks validity of inputs for Bernoulli Mixture Model
        '''
        X = check_array(X, accept_sparse = ['csr'])
        classes_ = _get_classes(X)
        n = len(classes_)
        
        # check that there are only two categories in data
        if n != 2:
           raise ValueError(('There are {0} categorical values in data, '
                             'should be only 2'.format(n)))
        
        # check that input data consists of only 0s and 1s
        if not 0 in classes_ or not 1 in classes_:
            raise ValueError(('Input data for Mixture of Bernoullis should consist'
                              'of zeros and ones, observed classes are {0}').format(classes_))
        try:
            check_is_fitted(self, 'means_')
        except:
            self.classes_ = classes_
        return X
        
        
        

class GaussianMixture(object):
    
    
    def _init_params(self,X):
        '''
        Initialise parameters
        '''
        d = X.shape[1]

        # initialise prior on means & precision matrices
        if 'means' in self.init_params:
            means0   = self.init_params['means']
        else:
            kms = KMeans(n_init = self.n_init, n_clusters = self.n_components)
            means0 = kms.fit(X).cluster_centers_
            
        if 'covar' in self.init_params:
            scale_inv0 = self.init_params['covar']
            scale0     = pinvh(scale_inv0)
        else:
            # heuristics to define broad prior over precision matrix
            diag_els   = np.abs(np.max(X,0) - np.min(X,0))/2
            scale_inv0 = np.diag( diag_els  )
            scale0     = np.diag( 1./ diag_els )
            
        if 'weights' in self.init_params:
            weights0  = np.ones(self.n_components) / self.n_components
        else:
            weights0  = np.ones(self.n_components) / self.n_components
          
        if 'dof' in self.init_params:
            dof0 = self.init_params['dof']
        else:
            dof0 = d
            
        if 'beta' in self.init_params:
            beta0 = self.init_params['beta']
        else:
            beta0 = 1e-3
            
        # clusters that are not pruned 
        self.active  = np.ones(self.n_components, dtype = np.bool)
        
        # checks initialisation errors in case parameters are user defined
        assert dof0 >= d,( 'Degrees of freedom should be larger than '
                                'dimensionality of data')
        assert means0.shape[0] == self.n_components,('Number of centrods defined should '
                                                     'be equal to number of components')
        assert means0.shape[1] == d,('Dimensioanlity of means and data '
                                          'should be the same')
        assert weights0.shape[0] == self.n_components,('Number of weights should be '
                                                           'to number of components')
        
        # At first iteration these parameters are equal to priors, but they change 
        # at each iteration of mean field approximation
        scale   = np.array([np.copy(scale0) for _ in range(self.n_components)])
        means   = np.copy(means0)
        weights = np.copy(weights0)
        dof     = dof0*np.ones(self.n_components)
        beta    = beta0*np.ones(self.n_components)
        init_   = [means0, scale0, scale_inv0, beta0, dof0, weights0]
        iter_   = [means, scale, scale_inv0, beta, dof, weights]
        return init_, iter_
        
        
    def _check_X(self,X):
        '''
        checks validity of input
        '''
        X = check_array(X)
        return X
        
        
        
class PoissonMixture(object):
    
    def _init_params(self,X):
        pass
        
    def _check_X(self,X):
        pass
        
        
if __name__ == '__main__':
    x1 = np.zeros([5,2])
    x1[1,1] = 1
    x1[3,1] = 2
    bmm1 = BernoulliMixture()
    bmm1
    x = bmm1._check_X(x1)
        