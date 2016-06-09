from __future__ import division
import numpy as np
import time




if __name__=='__main__':
    import pyximport
    pyximport.install(setup_args={"include_dirs":np.get_include()},
                      reload_support=True)
    import gibbs_lda_cython
    words = np.random.randint(0,1000,10000000, dtype = np.int)
    docs  = np.random.randint(0,2000,10000000, dtype = np.int)
    ta = np.random.randint(0,5,10000000, dtype = np.int)
    n_words = 1000
    n_docs = 2000
    n_topics = 20
    print "Warning"
    gl = gibbs_lda_cython.GibbsLDA(n_topics)
    
    t1 = time.time()
    wt,dt= gibbs_lda_cython.word_doc_topic(words,docs,ta, n_docs, n_words,n_topics)
    t2 = time.time()
    print t2-t1
    
    