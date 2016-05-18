
if __name__ == "__main__":
    
    ## imports used in testing
    #import matplotlib.pyplot as plt
    ##
    # testing Bernoulli HMM
#    X = np.array([[0,0,0],[0,0,0],[0,0,0],[1,1,1],[1,1,1],[1,1,1],[0,0,0],[0,0,0],
#                  [0,0,0],[1,1,1],[1,1,1],[1,1,1],[0,0,0],[0,0,0],[0,0,0],[1,1,1],
#                  [1,1,1],[0,0,0],[0,0,0],[0,0,0],[1,1,1],[1,1,1],[1,1,1],[0,0,0],[0,0,0],
#                  [0,0,0],[1,1,1],[1,1,1],[1,1,1],[0,0,0],[0,0,0],[0,0,0],[1,1,1],
#                  [1,1,1]])
#
#         
#    bhmm = VBBernoulliHMM(n_iter = 100, verbose = True)
#    bhmm.fit(X,[17])
#    ## test filtering 
#    alpha = bhmm.predict(X)
#    prob = bhmm.filter(X)
#    probs = bhmm.predict_proba(X)
    #
    #
    #

    ## test viterbi    
    #log_pr_start, log_pr_trans, log_pr_x = bhmm._log_probs_params(bhmm._start_params_,bhmm._trans_params_,
    #                                              bhmm._emission_params_,X)
    #
    #best_states = bhmm.predict(X)
    #print best_states
    #
    
    
    # testing Gaussian HMM
    #X = np.random.random([200,2])
    #X[0:50,:] += 1000
    #X[100:150,:] +=1000
    #
    #ghmm = VBGaussianHMM(n_iter = 100, verbose = True)
    #ghmm.fit(X,[100])
    #alpha = ghmm.predict(X)
    #probs = ghmm.predict_proba(X)
    #filtered = ghmm.filter(X)
    
    
    # testing Multinoulli HMM
    X = np.array([['a','b'],['b','a'],['b','a'],['a','b'],['b','a'],['b','a'],['a','b'],
         ['b','a'],['b','a'],['a','b'],['b','a'],['b','a'],['a','b'],['b','a'],['b','a'],['a','b'],
         ['b','a'],['b','a'],['a','b'],['b','a'],['b','a'],['a','b'],['b','a'],['b','a'],['a','b'],
         ['b','a'],['b','a'],['a','b'],['b','a'],['b','a'],['a','b'],['b','a'],['b','a'],['a','b'],
         ['b','a'],['b','a'],['a','b'],['b','a'],['b','a'],['a','b'],['b','a'],['b','a'],['a','b']])
        
    bmhmm = VBMultinoulliHMM(alpha_emission = 2, verbose = True)
    bmhmm.fit(X,[15])
    alpha = bmhmm.filter(X)
    clusters = bmhmm.predict(X)
    probs = bmhmm.predict_proba(X)