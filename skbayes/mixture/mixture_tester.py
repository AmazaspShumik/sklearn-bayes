# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 00:54:47 2016

@author: amazaspshaumyan
"""


if __name__ == "__main__":
    X = np.ones([300,4])
    X[0:100,0]   = 0
    X[100:200,1] = 0
    X[200:300,2] = 0
    
    bmm = VBBMM(n_components = 3, n_iter = 100, alpha0 = 10, n_init = 5,
                compute_score = True, c = 1, d = 1, verbose = False)
    bmm.fit(csr_matrix(X))
#    cluster = bmm.predict(X)
#    resps   = bmm.predict_proba(X)
#    scores  = bmm.score(X)
#    print cluster
#    print "\n"
    #print np.exp(scores)
#    
#    # -------  Example with kaggle digit dataset ----------
    import pandas as pd
    import time
    
    Data = pd.read_csv('train.csv')
    data = Data[Data['label']<=10]
#    X    = np.array(data[data.columns[1:]])
#    X[(X>0)*(X<50)] = 25
#    X[(X >= 50)*(X < 150)] = 100
#    X[(X >= 150)*(X < 250)] = 200
#    X[(X >= 250)] = 255
    
    t1 = time.time()
    x = csr_matrix(X)
    t2 = time.time()
    print t2-t1    
    X[X>0] = 10
    #x    = csr_matrix(X)
    x = csr_matrix(X)
    bmm = VBBMM(n_components = 10, n_iter = 100, alpha0 = 10, n_init = 2,
                compute_score = False, c = 1, d = 1, verbose = True)
    t1 = time.time()
    bmm.fit(x)
    t2 = time.time()
    print t2-t1
#
####################
    
#    X = np.array([ [1,2,3],
#                   [1,2,3],
#                   [1,2,3],
#                   [3,2,1],
#                   [3,2,1],
#                   [3,2,1] ])
##               
#    mmm = VBMMM(n_components = 2, n_iter = 100, alpha0 = 100, compute_score = True,
#                verbose = True, beta0 = 1)
#    mmm.fit(X)
#    
#    def cluster_prototype(obj):
#        prototypes = [0]*obj.n_components
#        for k in range(obj.n_components):
#            prototypes[k] = obj.classes_[np.argmax(obj.means_[k],1)]
#        return prototypes
# 
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    
    def plotter(X, max_k,title, rand_state = 1, prune_thresh = 0.02, mfa_max_iter = 10):
        '''
        Plotting function for VBGMMARD clustering
        
        Parameters:
        -----------
        X: numpy array of size [n_samples, n_features]
           Data matrix
           
        max_k: int
           Maximum number of components
           
        title: str
           Title of the plot
           
        Returns:
        --------
        :instance of VBGMMARD class 
        '''
        # fit model & get parameters
        gmm = VBGMMARD(n_iter = 150, n_components = max_k, n_mfa_iter = mfa_max_iter,
                       prune_thresh = prune_thresh)
        gmm.fit(X)
        centers = gmm.means_
        covars  = gmm.covars_
        k_selected = centers.shape[0]
        
        # plot data
        fig, ax = plt.subplots(figsize = (10,6))
        ax.plot(X[:,0],X[:,1],'bo', label = 'data')
        ax.plot(centers[:,0],centers[:,1],'rD', markersize = 8, label = 'means')
        for i in range(k_selected):
            plot_cov_ellipse(pos = centers[i,:], cov = covars[i], ax = ax)
        
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend(loc = 2)
        plt.title((title + ', {0} initial clusters, {1} selected clusters').format(max_k,k_selected))
        plt.show()
        return gmm
        
        
    # plot_cov_ellipse function is taken from  
    # https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
    
    def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
        """
        Plots an `nstd` sigma error ellipse based on the specified covariance
        matrix (`cov`). Additional keyword arguments are passed on to the 
        ellipse patch artist.
    
        Parameters
        ----------
            cov : The 2x2 covariance matrix to base the ellipse on
            pos : The location of the center of the ellipse. Expects a 2-element
                sequence of [x0, y0].
            nstd : The radius of the ellipse in numbers of standard deviations.
                Defaults to 2 standard deviations.
            ax : The axis that the ellipse will be plotted on. Defaults to the 
                current axis.
            Additional keyword arguments are pass on to the ellipse patch.
    
        Returns
        -------
            A matplotlib ellipse artist
        """
        def eigsorted(cov):
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            return vals[order], vecs[:,order]
    
        if ax is None:
            ax = plt.gca()
    
        vals, vecs = eigsorted(cov)
        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    
        # Width and height are "full" widths, not radius
        width, height = 2 * nstd * np.sqrt(vals)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, fill = False,
                        edgecolor = 'k',linewidth = 4,**kwargs)
    
        ax.add_artist(ellip)
        return ellip
    
    
    X = np.zeros([600,2])
    X[0:200,:]   = np.random.multivariate_normal(mean = (0,7), cov = [[1,0],[0,1]], size = 200)
    X[200:400,:] = np.random.multivariate_normal(mean = (0,0) , cov = [[1,0],[0,1]], size = 200)
    X[400:600,:] = np.random.multivariate_normal(mean = (0,-7) , cov = [[1,0],[0,1]], size = 200)
    
    sy_gmm_1 = plotter(X,20,'Synthetic Example')
    gmm = VBGMMARD(n_components = 3, n_iter = 10, verbose = True)
    gmm.fit(X)