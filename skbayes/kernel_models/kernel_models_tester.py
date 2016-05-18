if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.utils.estimator_checks import check_estimator
    from sklearn.cross_validation import train_test_split
    import time
    X = np.zeros([100000,1])
    X[:,0] = np.linspace(-2,2,100000)
    Y = np.sinc(X[:,0])
    X,x,Y,y = train_test_split(X,Y, test_size = 0.99)    
    klr =KernelisedElasticNetRegression(kernel = 'rbf', degree =2, alpha = 0.1)
    
    
    klr.fit(X,Y)    
    t1 = time.time()
    y_hat = klr.predict(x)
    t2 = time.time()
    print "Kernelised version {0}".format(t2- t1)
    
    enet = ElasticNet(alpha = 0.1)
    
    K = get_kernel(X,X,gamma=1, degree=2, coef0=1, kernel = 'rbf', kernel_params = None) 
    enet.fit(K,Y) 
    t1 = time.time()  
    k = get_kernel(x,X,gamma=1, degree=2, coef0=1, kernel = 'rbf', kernel_params = None)
    y_hat = enet.predict(k)
    t2 = time.time()
    print " Not Kernelised version {0}".format(t2- t1)
    
    

    plt.plot(x[:,0],y_hat,'ro')
    plt.plot(x[:,0],y,'b+')
    plt.show()
    
    check_estimator(KernelisedElasticNetRegression)
    check_estimator(KernelisedLassoRegression)
        
        
    #  CLASSIFICATION
    from sklearn.datasets import make_moons

    # Parameters
    n = 500
    test_proportion = 0.2

    # create dataset & split into train/test parts
    Xx,Yy   = make_moons(n_samples = n, noise = 0.3, random_state = 1)
    X,x,Y,y = train_test_split(Xx,Yy,test_size = test_proportion, 
                                 random_state = 2)
    klr = KernelisedLogisticRegressionL1()
    klr.fit(X,Y)
    
    # CLASSIFICATION SEVERAL CLASSES
    
    from sklearn.datasets import make_blobs
    centers = [(-3, -3), (0, 0), (3, 3)]
    n_samples = 600

    X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
                  centers=centers, shuffle=False, random_state=42)
    X, x, Y, y = train_test_split(X, y, test_size=0.1, random_state=42)
    klr = KernelisedLogisticRegressionL1(C= 0.1)
    klr.fit(X,Y)  
    print klr.predict(x)
    print y==klr.predict(x)  
    
    check_estimator(KernelisedLogisticRegressionL1)             
    from sklearn.utils import shuffle
    from sklearn.preprocessing import StandardScaler
    X_m, y_m = make_blobs(n_samples=300, random_state=0)
    X_m, y_m = shuffle(X_m, y_m, random_state=7)
    X_m = StandardScaler().fit_transform(X_m)
    # generate binary problem from multi-class one
    y_b = y_m[y_m != 2]
    X_b = X_m[y_m != 2]
    for (X, y) in [(X_m, y_m), (X_b, y_b)]:
        kl = KernelisedLogisticRegressionL1()
        kl.fit(X,y)
        y_hat = kl.predict(X)
        print float(np.sum(y_hat==y))/y.shape[0]
        if len(np.unique(y))==2:
            plt.plot(X[y==0,0],X[y==0,1],"ro")
            plt.plot(X[y==1,0],X[y==1,1],"bo")
            plt.show()
            d = kl.decision_function(X)
            print expit(d).shape