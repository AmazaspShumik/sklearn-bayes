from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.utils.validation import check_is_fitted,NotFittedError
from sklearn.utils import check_array
from scipy.sparse import csr_matrix,issparse,find
from scipy.misc import logsumexp
from scipy.special import gammaln
from time import time
import numpy as np
cimport cython
cimport numpy as np
import warnings
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

    alpha: float, optional (DEFAULT = 1)
        Concentration parameter for Dirichlet prior on topic distribution
        
    gamma: float, optional (DEFAULT = 1)
        Concentration parameter for Dirichlet prior on word distribution
                             
    compute_score: bool, optional (DEFAULT = False)
       If True computes joint log likelihood
       
    log_scale: bool, optional (DEFAULT = False)
       If True sampler will use log-scale (NOTE!!! This makes it longer to take sample,
       but process is)
       
       
    Attributes
    ----------
    word_topic_: numpy array of size (n_topics,n_words)
        Topic word distribution, self.components_[i,j] - number of times word j
        was assigned to topic i
         
    doc_topic_: numpy array of size (n_docs,n_topics) 
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
    def __init__(self, n_topics, n_burnin=30, n_thin=3, alpha=1, gamma=1,
                 compute_score=False, verbose=False):
        self.n_topics      = n_topics
        self.n_burnin      = n_burnin
        self.n_thin        = n_thin
        self.alpha         = alpha
        self.gamma         = gamma
        self.compute_score = compute_score
        self.scores_       = []
        self.verbose       = verbose

                
    def _init_params(self,X):
        '''
        Initialise parameters
        '''
        # parameters of Dirichlet priors for topic & word distribution
        topic_assignment = 0
        if self.alpha <= 0:
            raise ValueError(('alpha should be positive value, '
                              'observed {0}').format(self.alpha))
        if self.gamma <= 0:
            raise ValueError(('gamma should be positive value, '
                              'observed {0}').format(self.gamma))
        n_d = np.array(X.sum(1), dtype = np.int)
        corpus_size = np.sum(n_d)
        topic_assignment = np.random.randint(0,self.n_topics,corpus_size,dtype=np.int)
        return topic_assignment,n_d
        
        
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
        
        
        
    def fit_transform(self,X):
        '''
        Fit model and transform 
        
        Parameters
        ----------
        X: array-like or sparse matrix of size (n_docs,n_vocab)
           Document Word matrix
           ( Note we assume that there are no empty documents! )
           
        Returns
        -------
        dt: numpy array of size (n_docs,n_topics)
           Document topic matrix
        '''
        # run burn-in stage
        _ = self.fit(X)
        # make one more sample
        wt,dt,ta,ts = self._gibbs_sample_lda(self._words, self._docs, self._topic_assignment,
                                             self.word_topic_, self.doc_topic_, self.topics_, 
                                             self._tf, self._n_d, self._n_words) 
        empty_docs = self._n_d[:,0]==0
        dtd = np.array(dt,dtype = np.double)
        dtd[empty_docs,:] = 1.0 / self.n_topics
        dtd[~empty_docs,:] = dtd[~empty_docs,:] / self._n_d[~empty_docs,:]
        return dtd


    def fit(self,X):
        '''
        Runs burn-in stage of collapsed Gibbs sample for LDA model
        
        Parameters
        ----------
        X: array-like or sparse matrix of size (n_docs,n_vocab)
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
        topic_assignment, n_d = self._init_params(X)
        
        # compute initial word topic and document topic matrices
        word_topic,doc_topic,topics = word_doc_topic(words,docs,tf,topic_assignment,
                                                    n_docs,n_words, self.n_topics)                            
        # run burn-in samples
        for j in range(self.n_burnin):
            
             t0 = time()
             # one iteration of collapsed Gibbs Sampler 
             word_topic,doc_topic,topic_assignment,topics = self._gibbs_sample_lda(words,docs,
                                                            topic_assignment,word_topic,
                                                            doc_topic,topics,tf,n_d,n_words)             
             # compute joint log-likelihood if required
             if self.compute_score:
                 self.scores_.append(self._joint_loglike(n_d,n_words,n_docs,doc_topic,
                                                        word_topic,topics))
             # print info
             if self.verbose:
                 if not self.compute_score:
                     print( ("collected {0} sample in burn-in stage, "
                               "time spent on sample = {1}").format(j, time()-t0)) 
                 else:
                     print(("collected {0} sample in burn-in stage, "
                            "time spent on sample = {1}, log-like = {2}").format(j, time()-t0,
                             self.scores_[-1]))

        # save parameters from last sample of burn-in stage                                                                      
        self.word_topic_ = word_topic
        self.doc_topic_  = doc_topic
        self.topics_ = topics
        self._words  = words
        self._docs = docs
        self._tf = tf 
        self._topic_assignment = topic_assignment
        self._n_words = n_words
        self._n_d = n_d
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
        cdef int wi,di,ti,i,k,j
        cdef int cum_i = 0
        cdef np.ndarray[np.double_t,ndim=1] p_z
        cdef int n_topics = self.n_topics
        cdef double partial_sum
        
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
        
               # compute p(z_{n,d} = k| Z_{-n,d}) (i.e. probability of assigning
               # topic k for word n in document d, given all other topic assignments)
               p_z = (doc_topic[di] + alpha) / (alpha*n_topics + max(n_d[di,0] - 1,0) )  
                     
               # compute p(W|Z) (i.e. probability of observing corpus given all
               # topic assignments) and by multiplying it to p(z_{n,d} = k| Z_{-n,d})
               # obtain unnormalised p(z_{n,d}| DATA)
               p_z *= (word_topic[wi,:] + gamma) / (gamma*n_words + topics)
               
               # normalise & handle any conversion issues 
               normalizer = np.sum(p_z)
               partial_sum = 0.0
               for k in xrange(self.n_topics-1):
                   p_z[k] /= normalizer
                   partial_sum += p_z[k]
               p_z[n_topics-1] = 1.0 - partial_sum

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
        
        # log of normalization constant for prior of word distribution
        ll += self.n_topics*gammaln(n_words*self.gamma) - n_words*gammaln(self.gamma)

        # log of latent dist pdf without normalization constant (obtained after 
        # integrating out topic distribution)
        ll += np.sum(gammaln(self.alpha + doc_topic))
        ll -= np.sum(gammaln(self.n_topics*self.alpha + n_d[:,0]))

        # log p( words | latent_var), obtained after integrating out word
        # distribution
        ll += np.sum(gammaln(self.gamma + word_topic))
        ll -= np.sum(gammaln(n_words*self.gamma + topics))
           
        return ll
        
   
    @cython.wraparound(False)
    @cython.boundscheck(False)        
    def sample(self, n_samples = 5):
        '''
        Compute samples from posterior distribution
        
        Parameters
        ----------
        n_samples: int, optional (DEFAULT = 5)
             Number of samples
        
        Returns
        -------
        samples: list of dictionaries, length = n_samples
            Each element of list is dictionary with following fields
            
            - word_topic: numpy array of size [n_words, n_topics]
                 word_topic[m,k] - number of times word m was assigned 
                 to topic k
                 
            - doc_topic: numpy array of size [n_docs, n_topics]
                 doc_topic[n,k] - number of words in document n, that are assigned 
                 to to topic k
                 
            - topics: numpy array of size (n_topics,)
                 topics[k] - number of words assigned to topic k 
        '''
        check_is_fitted(self,'_n_words')
        samples = []
        for i in xrange((n_samples-1)*self.n_thin + 1):
            wt,dt,ta,ts = self._gibbs_sample_lda(self._words, self._docs, self._topic_assignment,
                                                 self.word_topic_, self.doc_topic_, self.topics_, 
                                                 self._tf, self._n_d, self._n_words)
            if i%self.n_thin==0:
                samples.append({'word_topic':wt,'doc_topic':dt,'topics':ts})
        return samples


        
    def transform(self, X, n_iter = 5):
        '''
        Transforms data matrix X (finds lower dimensional representation)
        Parameters obtained during burn-in are used as starting point for running
        new chain in order to transform matrix X.
        NOTE!!! It is highly advised to use fit_transform method on whole dataset !!!
        
        Parameters
        ----------
        X: array-like or sparse matrix of size (n_docs,n_vocab)
           Document Word matrix
           
        n_iter: int, optional (DEFAULT = 5)
           Number of gibbs sample iterations
        
        Returns
        -------
        doc_topic: numpy array of size (n_docs,n_topics)
           Matrix of document topics
        '''
        X = self._check_X(X)
        docs,words,tf = vectorize(X)
        n_docs, n_words = X.shape
        n_d = np.array(X.sum(1))
        cdef np.ndarray[DTYPE_t, ndim=2] wt = np.zeros([n_words,self.n_topics],dtype = DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=2] dt = np.zeros([n_docs,self.n_topics], dtype = DTYPE)
        cdef np.ndarray[DTYPE_t, ndim=1] ta = np.zeros(np.sum(n_d), dtype = np.int)
        cdef int cumi = 0
        cdef int i,wi,di,j,ti
        for i in xrange(len(words)):
            wi = words[i]
            di = docs[i]
            ti = np.argmax(self.word_topic_[wi,:])
            for j in xrange(tf[i]):
                ta[cumi] = ti
                wt[wi,ti] += 1
                dt[di,ti] += 1
                cumi += 1
        docs  = np.array(docs,dtype = np.int)
        words = np.array(words,dtype = np.int)
        tf    = np.array(tf,dtype = np.int)
        for k in xrange(n_iter):
            wt,dt,ta,ts = self._gibbs_sample_lda( words, docs, ta, wt, dt, self.topics_, 
                                                  tf, n_d, n_words)
        empty_docs = n_d[:,0]==0
        dtd = np.array(dt,dtype = np.double)
        dtd[empty_docs,:] = 1.0 / self.n_topics
        dtd[~empty_docs,:] = dtd[~empty_docs,:] / n_d[~empty_docs,:]
        return dtd


