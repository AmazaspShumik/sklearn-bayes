from sklearn.base import BaseEstimator,TransformerMixin
from scipy.sparse import csr_matrix,issparse
import numpy as np
cimport cython
cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t


def safe_sparse_sum(X,axis = None):
    '''
    Sum of elements of matrix for sparse & dense arrays
    '''
    if issparse(X):
        return X.sum(axis)
    else:
        return np.sum(X,axis)


@cython.wraparound(False)                
@cython.boundscheck(False)
def word_doc_topic(np.ndarray[DTYPE_t, ndim=1] words, np.ndarray[DTYPE_t, ndim=1] docs,
                   np.ndarray[DTYPE_t, ndim=1] topic_assignment, int n_docs, int n_words,
                   int n_topics):
    '''
    Computes initial word topic matrix and document topic matrix
    '''
    # initialise word-topic & doc-topic 
    cdef np.ndarray[DTYPE_t, ndim=2] word_topic = np.zeros([n_words,n_topics],dtype = DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] doc_topic = np.zeros([n_docs,n_topics], dtype = DTYPE)
    cdef int i
    cdef DTYPE_t topic_id
    for i in xrange(len(topic_assignment)):
        topic_id = topic_assignment[i]
        word_topic[words[i],topic_id] += 1
        doc_topic[docs[i],topic_id] += 1
    return word_topic   
        
        
def gibbs_sample_lda(np.ndarray[DTYPE_t, ndim=2] f, np.ndarray[DTYPE_t, ndim=2] g):
    pass
    
    


class GibbsLDA(object):
    '''
    Collapsed Gibbs Sampler for Latent Dirichlet Allocation
    
    Parameters
    ----------
    n_topics: int
        Number of topics in corpus

    n_burnin: int, optional (DEFAULT = 30)
        Number of samples to train model (it is expected that chain will
        converge in n_burnin iterations)
        
    n_thin: int, optional (DEFAULT = 3)
        Number of iterators between samples (to avoid autocorrelation between
        consecutive samples), thining is implemented after burnin.
        
    init_params: dict or None, optional (DEFAULT = None)
        Dictionary containing all relevant parameters
        
        - alpha: float, optional (DEFAULT = 1)
                 concentration parameter for Dirichlet prior on topic distribution
        
        - gamma: float, optional (DEFAULT = 1)
                 concentration parameter for Dirichlet prior on word distribution
                 
        - topic_assignment: 
            
    compute_score: bool, optional (DEFAULT = False)
       If True computes joint log likelihood
    '''
    
    def __init__(self, n_topics, n_burnin = 30, n_thin = 3, init_params = None,
                 compute_score = False, verbose = False):
        self.n_topics      = n_topics
        self.n_burnin      = n_burnin
        self.n_thin        = n_thin
        self.init_parms    = init_params
        self.compute_score = compute_score
        self.scores_       = []
        self.verbose       = verbose

                
    def _init_params(self,X):
        ''' '''
        # parameters of Dirichlet priors for topic & word distribution
        alpha = 1
        gamma = 1
        topic_assignment = 0
        if 'alpha' in self.init_params:
            alpha = self.init_params['alpha']
            if alpha <= 0:
                raise ValueError(('alpha should be positive value, '
                                  'observed {0}').format(alpha))
        if 'gamma' in self.init_params:
            gamma = self.init_params['gamma']
        return alpha,gamma,topic_assignment
        
        
   
    def _word_doc_topic(self,np.ndarray[DTYPE_t, ndim=1] words, np.ndarray[DTYPE_t, ndim=1] docs,
                   np.ndarray[DTYPE_t, ndim=1] topic_assignment, int n_docs, int n_words,
                   int n_topics):
        # initialise word-topic & doc-topic 
        cdef np.ndarray[DTYPE_t, ndim=2] word_topic = np.zeros([n_words,n_topics],dtype = DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] doc_topic = np.zeros([n_docs,n_topics], dtype = DTYPE)
        cdef int i
        cdef DTYPE_t topic_id
        for i in xrange(len(topic_assignment)):
           topic_id = topic_assignment[i]
           word_topic[words[i],topic_id] += 1
           doc_topic[docs[i],topic_id] += 1
        return word_topic   


    def fit(self,X):
        '''
        Runs burnin stage of collapsed Gibbs sample for LDA model
        
        Parameters
        ----------
        X: sparse matrix of size (n_docs,n_vocab)
           Document Word matrix
        
        Returns
        -------
        obj: self
           self
        '''
        pass
        #X = check_array(X, accept_sparse = ['csr'])
        
        # number of words in each document
        #self.n_d = sparse_safe_sum(X,0)
        
        # initialise topic assignments & parameters of prior distribution
        #alpha,gamma,topic_assignment = self._init_params(X)
        
        #for j in range(self.n_burnin):
        #    pass
            
        #    # collapsed gibbs sample 
        #    #word_topic, doc_topic, topic_assignment = gibbs_sample_lda(X,word_topic,doc_topic,
        #    #                                                           topic_assignment)
                                                                       
        # save parameters from last  sample                                                                       
        #self.word_topic = word_topic
        #self.doc_topic  = doc_topic
        #self.topic_assignment = topic_assignment

    def samplefuck(self,n_samples = 1):
        '''
        Samples from posterior 
        
        Parameters
        ---------
        '''
        return 'fun'
        
    
    def transform(self,X):
        '''
        '''
        word_topic,doc_topic,topic_assignment = gibbs_sample_lda(X,self.word_topic,
                                                                 self.doc_topic,
                                                                 self.topic_assignment)
        return doc_topic
   
