from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.utils.validation import check_is_fitted,NotFittedError
from sklearn.utils import check_array
from scipy.sparse import csr_matrix,issparse,find
from scipy.misc import logsumexp
from scipy.special import gammaln
import numpy as np
cimport cython
cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t


@cython.wraparound(False)                
@cython.boundscheck(False)
def word_doc_topic(np.ndarray[DTYPE_t, ndim=1] words, np.ndarray[DTYPE_t, ndim=1] docs,
                   np.ndarray[DTYPE_t, ndim=1] tf, np.ndarray[DTYPE_t, ndim=1] topic_assignment,
                   int n_docs, int n_words, int n_topics):
    '''
    Computes initial word topic and document topic matrices
    '''
    cdef np.ndarray[DTYPE_t, ndim=2] word_topic = np.zeros([n_words,n_topics],dtype = DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] doc_topic = np.zeros([n_docs,n_topics], dtype = DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] topics = np.zeros(n_topics, dtype = DTYPE)
    cdef int corpus_id = 0
    cdef int i
    cdef DTYPE_t topic_id
    for i in xrange(len(tf)):
        for j in xrange(tf[i]):
            topic_id = topic_assignment[corpus_id]
            word_topic[words[i],topic_id] += 1
            doc_topic[docs[i],topic_id] += 1
            topics[topic_id] += 1
            corpus_id += 1
    return word_topic,doc_topic,topics
    


def vectorize(X):
    '''
    Vectorize document term matrix.
    '''
    if issparse(X):
        docs,words,tf = find(X)
    else:
        docs,words = np.where(X>0)
        tf  = X[docs,words]
    return docs,words,tf


        
class GibbsLDA(BaseEstimator,TransformerMixin):
    '''
    Collapsed Gibbs Sampler for Latent Dirichlet Allocation
    
    Parameters
    ----------
    n_topics: int
        Number of topics in corpus

    n_burnin: int, optional (DEFAULT = 30)
        Number of samples to train model (it is expected that chain will
        converge during burn-in stage)
        
    n_thin: int, optional (DEFAULT = 3)
        Number of iterators between samples (to avoid autocorrelation between
        consecutive samples), thining is implemented after burnin.
        
    init_params: dict, optional (DEFAULT = {})
        Dictionary containing all relevant parameters
        
        - alpha: float, optional (DEFAULT = 1)
                 concentration parameter for Dirichlet prior on topic distribution
        
        - gamma: float, optional (DEFAULT = 1)
                 concentration parameter for Dirichlet prior on word distribution
                 
        - topic_assignment: 
            
    compute_score: bool, optional (DEFAULT = False)
       If True computes joint log likelihood
       
       
    Attributes
    ----------
    components_: numpy array of size (n_topics,n_words)
        Topic word distribution, self.components_[i,j] - number of times word j
        was assigned to topic i
         
    doctopic_: numpy array of size (n_docs,n_topics) 
        Document topic distribution, self.doctopic_[i,j] - number of times topic
        j was observed in document i
        
    topics_: numpy array of size (n_topics,)
        Number of words assigned to each topic
    
    scores_: list of length n_burnin
        Values of joint log likelihood
    
        
    References
    -----------
    Griffiths and Steyers, Finding Scientific Topics (2004)
    K.Murphy, Machine Learning A Probabilistic Perspective (2012)
    '''
    def __init__(self, n_topics, n_burnin = 30, n_thin = 3, init_params = None,
                 compute_score = False, verbose = False):
        self.n_topics      = n_topics
        self.n_burnin      = n_burnin
        self.n_thin        = n_thin
        self.init_parms    = {} if init_params is None else init_params
        self.compute_score = compute_score
        self.scores_       = []
        self.verbose       = verbose
        self.init_params   = init_params

                
    def _init_params(self,X):
        '''
        Initialise parameters
        '''
        # parameters of Dirichlet priors for topic & word distribution
        alpha = 1; gamma = 1
        topic_assignment = 0
        if 'alpha' in self.init_params:
            alpha = self.init_params['alpha']
            if alpha <= 0:
                raise ValueError(('alpha should be positive value, '
                                  'observed {0}').format(alpha))
        if 'gamma' in self.init_params:
            gamma = self.init_params['gamma']
            if gamma <= 0:
                raise ValueError(('gamma should be positive value, '
                                  'observed {0}').format(gamma))
        n_d = X.sum(0)
        corpus_size = np.sum(n_d)
        topic_assignment = np.random.randint(0,self.n_topics,corpus_size,dtype=np.int)
        return alpha,gamma,topic_assignment,n_d
        
        
    def _check_X(self,X):
        '''
        Validate input matrix
        '''
        X = check_array(X, accept_sparse = ['csr'])
        
        # check that document term matrix is non negative
        arr = X.data if issparse(X) else X
        if np.sum(arr<0) > 0:
            raise ValueError('Document term matrix should not contain negative values')
        
        # if model was fitted before check that vocabulary size is the same
        if '_n_words' in dir(self):
            assert(X.shape[1] == self._n_words,("vocabulary size should be the "
                                                "same for train and test sets"))
        return X


    def fit(self,X):
        '''
        Runs burn-in stage of collapsed Gibbs sample for LDA model
        
        Parameters
        ----------
        X: sparse matrix of size (n_docs,n_vocab)
           Document Word matrix
        
        Returns
        -------
        obj: self
           self
        '''
        X = self._check_X(X)
        docs,words,tf = vectorize(X) # tf is term frequency
        docs  = np.array(docs,dtype = np.int)
        words = np.array(words,dtype = np.int)
        tf    = np.array(tf,dtype = np.int)
         
        #number of documents and size of vocabulary
        n_docs,n_words = X.shape

        # initialise topic assignments & parameters of prior distribution
        self.alpha, self.gamma, topic_assignment, n_d = self._init_params(X)
        
        # compute initial word topic and document topic matrices
        word_topic,doc_topic,topics = word_doc_topic(words,docs,tf,topic_assignment,
                                                    n_docs,n_words, self.n_topics) 
        # run burn-in samples
        for j in range(self.n_burnin):
            
             # one iteration of collapsed Gibbs Sampler 
             word_topic,doc_topic,topic_assignment,topics = self._gibbs_sample_lda(words,docs,
                                                            topic_assignment,word_topic,
                                                            doc_topic,topics,tf,n_d,n_words)
             
             # compute joint loh-likelihood if required
             if self.compute_score:
                 self.scores_.append(self._joint_loglike(n_d,n_words,n_docs,doc_topic,
                                                        word_topic,topics))

        # save parameters from last  sample                                                                       
        self.components_ = word_topic.T
        self.doctopic_  = doc_topic
        self._topic_assignment = topic_assignment
        self.topics_ = topics
        self._n_words = n_words
        return self
    


    @cython.wraparound(False)
    @cython.boundscheck(False)
    def _gibbs_sample_lda(self,np.ndarray[DTYPE_t,ndim=1] words, np.ndarray[DTYPE_t,ndim=1] docs,
                     np.ndarray[DTYPE_t,ndim=1] topic_assignment, np.ndarray[DTYPE_t, ndim=2] word_topic,
                     np.ndarray[DTYPE_t,ndim=2] doc_topic, np.ndarray[DTYPE_t, ndim=1] topics,
                     np.ndarray[DTYPE_t,ndim=1] tf, np.ndarray[DTYPE_t,ndim=2] n_d, int n_words):
        '''
        Collapsed Gibbs Sampler (single sample)
        '''
        cdef double alpha = self.alpha
        cdef double gamma = self.gamma
        cdef int wi,di,ti,i,j
        cdef int cum_i = 0
        cdef np.ndarray[np.double_t,ndim=1] logp_latent
        cdef np.ndarray[np.double_t,ndim=1] p_z
        cdef int n_topics = self.n_topics
        
        for i in xrange(len(words)):
            for j in xrange(tf[i]):
                
               # retrieve doc, word and topic indices
               wi = words[i]
               di = docs[i]
               ti = topic_assignment[cum_i]
        
               # remove all 'influence' of i-th word in corpus
               word_topic[wi,ti] -= 1
               doc_topic[di,ti] -= 1
               topics[ti] -= 1
        
               # compute log(p(z_{n,d} = k| Z_{-n,d})) (i.e. log probability of assigning
               # topic k for word n in document d, given all other topic assignments)
               logp_latent = np.log(doc_topic[di,:]+alpha) - np.log(alpha*n_topics + n_d[0,di]);
        
               # compute log(p(W|Z)) (i.e. log probability of observing corpus given all
               # topic assignments) and by adding it to log(p(z_{n,d} = k| Z_{-n,d}))
               # obtain unnormalised p(z_{n,d}| DATA)
               logp_latent += np.log(word_topic[wi,:] + gamma) - np.log(gamma*n_words + topics);
            
               # normalise move away from log scale
               logp_latent -= logsumexp(logp_latent)
               p_z = np.exp(logp_latent)
            
               # make sample from multinoulli distribution & update topic assignment
               ti = np.where(np.random.multinomial(1,p_z))[0][0]
               topic_assignment[cum_i] = ti
            
               # add 'influence' of i-th element in corpus back
               word_topic[wi,ti] += 1
               doc_topic[di,ti] += 1
               topics[ti] += 1
               cum_i += 1
            
        return word_topic, doc_topic, topic_assignment, topics
        
        
    def _joint_loglike(self,n_d,n_words,n_docs,doc_topic,word_topic,topics):
        '''
        Computes joint log likelihood of latent and observed variables
        '''
        ll = 0
        
        # log of normalization constant for prior of topic distrib
        ll += n_docs*(gammaln(self.n_topics*self.alpha) - self.n_topics*gammaln(self.alpha))

        # log of latent dist pdf without normalization constant (obtained after 
        # integrating out topic distribution)
        ll += np.sum(gammaln(self.alpha + doc_topic))
        ll -= np.sum(gammaln(self.n_topics*self.alpha + n_d))

        # log of normalization constant for prior of word distribution
        ll += self.n_topics*gammaln(n_vocab*self.gamma) - n_vocab*gammaln(self.gamma)

        # log p( words | latent_var), obtained after integrating out word
        # distribution
        ll += np.sum( gammaln(self.gamma + word_topic) )
        ll -= np.sum( gammaln(n_vocab*self.gamma + topics) )
        return ll
        
        
    def transform(self,X):
        '''
        Transforms data matrix X (finds lower dimensional representation)
        
        Parameters
        ----------
        X: sparse matrix of size (n_docs,n_vocab)
           Document Word matrix
        
        Returns
        -------
        doc_topic: numpy array of size (n_docs,n_topics)
           Matrix of document topics
        '''
        X = self._check_X(X)
        words,docs,n_d = vectorize(X)
        wt,doc_topic,ta,ts = self._gibbs_sample_lda(words, docs, self._topic_assignment,
                                                    self.components_.T, self.doctopic_,
                                                    self.topics_, n_d, self._n_words)
        return doc_topic

        
        
        
        
    
        
        
    
            
        
    