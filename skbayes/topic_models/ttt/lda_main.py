from __future__ import division
import numpy as np




if __name__=='__main__':
    #import pyximport
    #pyximport.install(setup_args={"include_dirs":np.get_include()},
    #                  reload_support=True)
    import gibbs_lda
    words = np.array([0,1,2,3,4,5,1,2,0,4,5], dtype = np.int)
    docs  = np.array([0,1,1,0,1,1,2,0,2,2,2], dtype = np.int)
    ta = np.array([0,0,1,1,0,0,1,1,1,0], dtype = np.int)
    n_words = 6
    n_docs = 3
    n_topics = 2
    gl = gibbs_lda.GibbsLDA(n_topics)
    print dir(gl)
    wt= gl.word_doc_topic(words,docs,ta, n_docs, n_words,n_topics)
       