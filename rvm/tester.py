# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 18:43:40 2016

@author: amazaspshaumyan
"""

        
    
    
if __name__ == "__main__":
    from sklearn.cross_validation import train_test_split
    from fast_scikit_rvm import RVR,RegressionARD,ClassificationARD,RVC
    import time
    
    
    
    ######################## TESTS for ARD REgression    ######################
#
#    n_features = 180
#    n_samples  = 400
#    X      = np.random.random([n_samples,n_features])
#    X[:,0] = np.linspace(0,10,n_samples)
#    Y      = 20*X[:,0] + 5 + np.random.normal(0,1,n_samples)
#    X,x,Y,y = train_test_split(X,Y,test_size = 0.4)
#    
#    # RegressionARD
#    ard = RegressionARD(n_iter = 200, verbose = True)
#    start_ard = time.time()
#    ard.fit(X,Y)
#    end_ard   = time.time()
#    y_hat = ard.predict(x)
#    ard_time = end_ard - start_ard
#    
#    # sklearn ARD
#    skard = ARDRegression()
#    start_skard = time.time()
#    skard.fit(X,Y)
#    end_skard   = time.time()
#    ysk_hat = skard.predict(x)
#    sk_time = end_skard - start_skard
#    
#    print "FAST BAYESIAN LEARNER"
#    print np.sum( (y - y_hat)**2 ) / n_samples
#    print "VARIATIONAL ARD"
#    print np.sum( (y - ysk_hat)**2 ) / n_samples
#    
#    import matplotlib.pyplot as plt
#    plt.plot(x[:,1],y_hat,'ro')
#    plt.plot(x[:,1],y,'b+')
#    plt.show()
#    
#    plt.plot(y_hat - y,'go')
#    plt.plot(ysk_hat - y,'ro')
#    plt.show()
#    
#    print 'timing sklearn {0}'.format(sk_time)
#    print 'timing ard sbl {0}'.format(ard_time)
#    
#    from sklearn.utils.testing import assert_array_almost_equal
#    def test_toy_ard_object():
#        # Test BayesianRegression ARD classifier
#        X = np.array([[1], [2], [3]])
#        Y = np.array([1, 2, 3])
#        clf = RegressionARD(compute_score=True)
#        clf.fit(X, Y)
#    
#        # Check that the model could approximately learn the identity function
#        test = [[1], [3], [4]]
#        assert_array_almost_equal(clf.predict(test), [1, 3, 4], 2)
#        
#    test_toy_ard_object()
    
    
#    from scipy import stats
#    ###############################################################################
#    # Generating simulated data with Gaussian weights
#    import rvm_fast
#
#    # Parameters of the example
#    n_samples, n_features = 1000, 800
#    # Create Gaussian data
#    X = np.random.randn(n_samples, n_features)
#    # Create weigts with a precision lambda_ of 4.
#    lambda_ = 4.
#    w = np.zeros(n_features)
#    # Only keep 10 weights of interest
#    relevant_features = np.random.randint(0, n_features, 10)
#    for i in relevant_features:
#        w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
#    # Create noite with a precision alpha of 50.
#    alpha_ = 1.
#    noise = stats.norm.rvs(loc=0, scale=1 / np.sqrt(alpha_) , size=n_samples)
#    # Create the target
#    y = np.dot(X, w) + noise
#    X,x,Y,y = train_test_split(X,y, test_size = 0.2)
#    
#    # sklearn ARD
#    skard = ARDRegression()
#    start_skard = time.time()
#    skard.fit(X,Y)
#    end_skard   = time.time()
#    ysk_hat = skard.predict(x)
#    sk_time = end_skard - start_skard
#    
#    
#    # RegressionARD blazing fast
#    ard = RegressionARD(fit_intercept = True, n_iter = 300, verbose = True)
#    start_ard = time.time()
#    ard.fit(X,Y)
#    end_ard   = time.time()
#    y_hat,var_hat = ard.predict_dist(x)
#    ard_time = end_ard - start_ard
#    
#    # just fast
#    ard1 = rvm_fast.RegressionARD(fit_intercept = True, n_iter = 300)
#    start_ard = time.time()
#    ard1.fit(X,Y)
#    end_ard   = time.time()
#    y_hat1 = ard1.predict(x)
#    ard1_time = end_ard - start_ard
#    
#    print "BlAZING FAST ARD"
#    print np.sum( ( y - y_hat )**2 ) / n_samples
#    print 'FAST ARD'
#    print np.sum( ( y - y_hat1 )**2 ) / n_samples
#    print "VARIATIONAL ARD"
#    print np.sum( ( y - ysk_hat )**2 ) / n_samples
#    print 'timing ard blazing {0}, features {1}'.format(ard_time,np.sum(ard.coef_!=0))
#    print 'timing ard fast {0}, features {1}'.format(ard1_time,np.sum(ard1.coef_!=0))
#    print 'timing sklearn {0}, features {1}'.format(sk_time,np.sum(skard.coef_!=0))

    from scipy import stats
    
    
    
    
    
    
#    # Parameters of the example
#    n_samples, n_features = 200, 150
#    # Create Gaussian data
#    X = np.random.randn(n_samples, n_features)
#    # Create weigts with a precision lambda_ of 4.
#    lambda_ = 0.01
#    w = np.zeros(n_features)
#    # Only keep 10 weights of interest
#    relevant_features = np.random.randint(0, n_features, 10)
#    for i in relevant_features:
#        w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
#    # Create noite with a precision alpha of 50.
#    # Create the target
#    y = np.dot(X, w) + 100
#    y_hat  = np.ones(y.shape[0])
#    y_hat[y < 98] = 0
#    X,x,Y,y = train_test_split(X,y_hat, test_size = 0.2)
#
#    
#    clf = ClassificationARD(normalize = False, n_iter = 200, tol_solver = 1e-5,
#                            penalise_intercept = False)
#    t1 = time.time()
#    clf.fit(X,Y)
#    t2 = time.time()
#    y_hat = clf.predict(x)
#    pr    = clf.predict_proba(x)
#    #y_hat = np.zeros(y.shape[0])
#    #y_hat[pr>0.5] = 1
#    print 'ERRor ARD'
#    print float(np.sum(y_hat!=y)) / y.shape[0]
#    print 'time ard {0}'.format(t2-t1)
#    
#    from sklearn.linear_model import LogisticRegression
#    lr = LogisticRegression(C = 1)
#    t1 = time.time()
#
#    lr.fit(X,Y)
#    t2 = time.time()
#    y_lr = lr.predict(x)
#    print 'error log reg'
#    print float(np.sum(y_lr!=y)) / y.shape[0]
#    print 'time lr {0}'.format(t2-t1)
#    plt.plot(w,'b-');plt.plot(clf.coef_[0][0,:],'ro')
    
#    
#    X_m, y_m = make_blobs(n_samples=300, random_state=0)
#    X_m, y_m = shuffle(X_m, y_m, random_state=7)
#    X_m = StandardScaler().fit_transform(X_m)
#    # generate binary problem from multi-class one
#    y_b = y_m[y_m != 2]
#    X_b = X_m[y_m != 2]
#    
#    clf = RVC()
#    clf.fit(X_b,y_b)
#    print "ERROR"
#    print np.sum(y_b != clf.predict(X_b))
    #plt.plot(lr.coef_[0,:],'go')
    
#    
#    
#    
    
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.cross_validation import train_test_split
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import mean_squared_error
    
    # parameters
    n = 20000
    
    # generate data set
    #np.random.seed(0)
    Xc       = np.ones([n,1])
    Xc[:,0]  = np.linspace(-5,5,n)
    Yc       = 10*np.sinc(Xc[:,0]) + np.random.normal(0,1,n)
    X,x,Y,y  = train_test_split(Xc,Yc,test_size = 0.5, random_state = 0)
    
    # train rvm with fixed-point optimization
    rvm = RVR(gamma = 1, coef0 =  0.1, kernel = 'rbf')
    t1 = time.time()
    rvm.fit(X,Y)
    t2 = time.time()
    y_hat,var     = rvm.predict_dist(x)
    rvm_err   = mean_squared_error(y_hat,y)
    rvs       = np.sum(rvm.active_)
    print "RVM error on test set is {0}, number of relevant vectors is {1}, time {2}".format(rvm_err, rvs, t2 - t1)
    from sklearn.svm import SVR
    from sklearn.grid_search import GridSearchCV
    svr = GridSearchCV(SVR(gamma = 1), param_grid = {'C':[0.001,0.1,1,10,100]}, cv = 5)
    t1 = time.time()
    svr.fit(X,Y)
    t2 = time.time()
    svm_err = mean_squared_error(svr.predict(x),y)
    svs     = svr.best_estimator_.support_vectors_.shape[0]
    print "SVM error on test set is {0}, number of support vectors is {1}, time {2}".format(svm_err, svs, t2 - t1)
    
    
    # plot test vs predicted data
    plt.figure(figsize = (12,8))
    plt.plot(x[:,0],y,"b+",markersize = 3, label = "test data")
    plt.plot(x[:,0],y_hat,"rD", markersize = 3, label = "mean of predictive distribution")
    # plot one standard deviation bounds
    plt.plot(x[:,0],y_hat + np.sqrt(var),"co", markersize = 3, label = "y_hat +- std")
    plt.plot(x[:,0],y_hat - np.sqrt(var),"co", markersize = 3)
    plt.plot(rvm.relevant_vectors_,Y[rvm.active_],"co",markersize = 11,  label = "relevant vectors")
    plt.legend()
    plt.title("RVM")
    plt.show()
    
#    ##########################################################
#    
#    n_iter = 100
#    from sklearn.svm import SVR
#    from sklearn.preprocessing import scale
#    
#    
#    def compare_rvr_svr(X,Y,kernel, gamma, coef0, degree, test_size):
#        '''
#        Compares perfomance of RVR and SVR
#        
#        #TODO: use timeit for timing
#        '''
#        X,x,Y,y = train_test_split(X,Y,test_size = test_size)
#        # RVR
#        rvm = RVR(gamma = gamma, coef0 = coef0, degree = degree, kernel = kernel)
#        t1 = time.time()
#        rvm.fit(X,Y)
#        t2 = time.time()
#        rvr_time = t2 - t1
#        y_hat = rvm.predict(x)
#        rvm_err   = mean_squared_error(y_hat,y)
#        rvs       = np.sum(rvm.active_)
#        # SVR
#        svm = SVR(gamma = gamma, coef0 = coef0, degree = degree, kernel = kernel)
#        svr = GridSearchCV(svm,param_grid = {'C':[0.01,0.1,1,10,100]}, cv = 5)
#        t1 = time.time()
#        svr.fit(X,Y)
#        t2 = time.time()
#        svm_time = t2 - t1
#        svm_err = mean_squared_error(svr.predict(x),y)
#        svs     = svr.best_estimator_.support_vectors_.shape[0]
#        return {'RVR':[rvm_err,rvr_time,rvs],
#                'SVR':[svm_err,svm_time,svs]}
#                
#        
#    from sklearn.datasets import load_boston
#    boston = load_boston()
#    Xb,yb  = scale(boston['data']),boston['target']
#    
#    from sklearn.datasets import load_diabetes
#    diabetes = load_diabetes()
#    Xd,yd  = scale(diabetes['data']),diabetes['target']
#    
#    
#    #b = compare_rvr_svr(Xb,yb,kernel = 'poly', gamma = 1, coef0 = 1, degree=3,
#    #                   test_size = 0.2)
#                        
#    d =  compare_rvr_svr(Xb,yb,kernel = 'rbf', gamma = 0.1, coef0 = 1, degree=2,
#                        test_size = 0.1)
    
    
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(RegressionARD)
    print 'Regression ARD passed tests'
    check_estimator(RVR)
    print 'RVR passed tests'
    check_estimator(ClassificationARD)
    print 'ClassificationARD passed'
    check_estimator(RVC)
    print "RVC passes"
    X, y = make_blobs(n_samples=30, random_state=0, cluster_std=0.1)
    X, y = shuffle(X, y, random_state=7)
    X = StandardScaler().fit_transform(X)
    # We need to make sure that we have non negative data, for things
    # like NMF
    X -= X.min() - .1
    y_names = np.array(["one", "two", "three"])[y]
    clf = ClassificationARD()
    clf.fit(X,y)
    y_hat = clf.predict(X)
    print 'estimator'
    print np.sum(y_hat!=y)
    
    plt.plot(X[y==2,0],X[y==2,1],'ro')
    plt.plot(X[y==1,0],X[y==1,1],'bo')
    plt.xlim(-5,5)
    plt.ylim(-5,5)
    
    
    import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ARDRegression
from sklearn.cross_validation import train_test_split
from scipy import stats
import time
%matplotlib inline



def single_simulation(n_samples = 100, n_features = 100, alpha_ = 50., 
                               lambda_ = 4., test_size = 0.2):
    '''
    Single simulation based on scikit-learn example`
    '''
    ###  Generate data (the same example as on scikit-learn website)
    
    # Create Gaussian data
    X = np.random.randn(n_samples, n_features)
    # Create weigts with a precision lambda_
    w = np.zeros(n_features)
    # Only keep 10 weights of interest
    relevant_features = np.random.randint(0, n_features, 10)
    for i in relevant_features:
        w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
    # Create noiee with a precision alpha 
    noise = stats.norm.rvs(loc=0, scale=1 / np.sqrt(alpha_) , size=n_samples)
    # Create the target
    y = np.dot(X, w) + noise
    X,x,Y,y = train_test_split(X,y, test_size = test_size)
    
    ###  Fit models & time 
    
    # sklearn ARD
    skard = ARDRegression()
    start_skard = time.time()
    skard.fit(X,Y)
    end_skard   = time.time()
    ysk_hat = skard.predict(x)
    sk_time = end_skard - start_skard
    
    # RegressionARD
    ard = RegressionARD(perfect_fit_tol = 1e-5, tol = 1e-4, n_iter = 100)
    start_ard = time.time()
    ard.fit(X,Y)
    end_ard   = time.time()
    y_hat = ard.predict(x)
    ard_time = end_ard - start_ard
    
    mse_ard =  np.sum( ( y - y_hat )**2 ) / n_samples
    mse_skard =  np.sum( ( y - ysk_hat )**2 ) / n_samples
    return {'sklearn':[mse_skard, sk_time],
            'sequential':[mse_ard, ard_time]}
    
    
    
    def simulation(n_iter = 100,n_samples = 100, n_features = 100, 
                   alpha_ = 50, lambda_ = 4, test_size = 0.2):
        skard_timer, ard_timer = [],[]
        skard_mse, ard_mse     = [],[]
        for i in range(n_iter):
           out = single_simulation(n_samples,n_features,alpha_, lambda_)
           skard_timer.append(out['sklearn'][1])
           skard_mse.append(out['sklearn'][0])
           ard_timer.append(out['sequential'][1])
           ard_mse.append(out['sequential'][0])
        statistics = [ard_mse,ard_timer,skard_mse,skard_timer]
        return [np.mean(s) for s in statistics]
    
    
    skard_timer, ard_timer = [],[]
    skard_mse, ard_mse     = [],[]
    
    #PARAMETERS
    N = 500
    M = [200,300,400,500,600,700,800,900,100]
    
    
    for m in M:
        am,at,skmse,skt = simulation(n_iter = 10, n_samples = N, n_features = m,  alpha_ = .2, lambda_ = 4. )
        skard_timer.append(skt), ard_timer.append(at)
        skard_mse.append(skmse), ard_mse.append(am)
        
    plt.plot(M,ard_mse,'g-', label = 'sequential ARD')
    plt.plot(M,skard_mse,'r-', label = 'sklearn ARD')
    plt.title('Comparison of MSE on test set')
    plt.xlabel('n_features')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    plt.plot(M,ard_timer,'g-', label = 'sequential ARD')
    plt.plot(M,skard_timer,'r-', label = 'sklearn ARD')
    plt.title('Timing')
    plt.xlabel('n_features')
    plt.ylabel('Comparison of fitting time')
    plt.legend()
    plt.show()
    
    
    
    

def _compare(X,Y,kernel, gamma, degree, test_size,
             cost, coef0, learn_type = 'regression'):
    '''
    Compares perfomance of RVR and SVR
    
    #TODO: use timeit for timing
    '''
    X,x,Y,y = train_test_split(X,Y,test_size = test_size)
    if learn_type == 'regression':
        RVM = RVR(perfect_fit_tol = 1e-5,tol = 1e-3)
        SVM = SVR()
    else:
        RVM = RVC()
        SVM = SVC()
    rvm = GridSearchCV(RVM,param_grid = {"gamma":gamma,'coef0':coef0,'kernel':kernel,
                                         'degree':degree})
    svm = GridSearchCV(SVM,param_grid = {"C":cost,"gamma":gamma,'coef0':coef0,'kernel':kernel,
                                         'degree':degree})
    
    # fit rvm
    t1 = time.time()
    rvm.fit(X,Y)
    t2 = time.time()
    rvm_time = t2 - t1
    y_rvm = rvm.predict(x)
    
    # fit svm
    t1 = time.time()
    svm.fit(X,Y)
    t2 = time.time()
    svm_time = t2 - t1
    y_svm = svm.predict(x)
    
    # calculate error on test set
    if learn_type == 'regression':
        rvm_err = mean_squared_error(y_rvm,y)
        svm_err = mean_squared_error(y_svm,y)
    else:
        rvm_err = float(np.sum(y_rvm!=y)) / y.shape[0]
        svm_err = float(np.sum(y_svm!=y)) / y.shape[0]
    return {'RVM':[rvm_err,rvm_time],
            'SVM':[svm_err,svm_time]}

# Parameters of simulation 


def compare_rvm_svm(X,Y,kernel, gamma, degree = 2,  test_size = 0.2,
                    cost = [0.01,0.1,1,10,100], coef0 = [0.01],
                    learn_type = 'regression', n_iter=1):
    error_svm,time_svm = [0]*n_iter,[0]*n_iter
    error_rvm,time_rvm = [0]*n_iter,[0]*n_iter
    for i in range(n_iter):
        out = _compare(X,Y,kernel, gamma, degree, test_size,
                      cost,coef0,learn_type = learn_type)
        error_svm[i],time_svm[i] = out["SVM"]
        error_rvm[i],time_rvm[i] = out["RVM"]
    return error_rvm,time_rvm,error_svm,time_svm
        

# import datasets  & scale  





# comparison
#out_boston   =  compare_rvm_svm(Xb,yb,kernel = ['rbf','sigmoid','poly'], 
#                                degree = [2,3],gamma = [0.1,1,10], test_size = 0.2,
#                                n_iter = 10)
out_diabetes =  compare_rvm_svm(Xd,yd,kernel = ['rbf','sigmoid','poly'], degree = [2,3],
                                gamma = [0.1,1,10], test_size = 0.1, n_iter = 1)


        

if __name__ == '__main__':
    import matplotlib.pyplot as plt     
    # create data set 
    x          = np.zeros([500,2])
    x[:,0]     = np.random.normal(0,1,500) 
    x[:,1]     = np.random.normal(0,1,500) 
    x[0:250,0] = x[0:250,0] + 5
    x[0:250,1] = x[0:250,1] + 10
    #x          = x - np.mean(x,0)
    #x          = scale(x)
    y          = -1*np.ones(500)
    y[0:250]   = 1
    blr        = BayesianLogisticRegression(solver = 'newton_cg')
    blr.fit(x,y)
    
    # create grid for heatmap
    n_grid = 500
    max_x      = np.max(x,axis = 0)
    min_x      = np.min(x,axis = 0)
    X1         = np.linspace(min_x[0],max_x[0],n_grid)
    X2         = np.linspace(min_x[1],max_x[1],n_grid)
    x1,x2      = np.meshgrid(X1,X2)
    Xgrid      = np.zeros([n_grid**2,2])
    Xgrid[:,0] = np.reshape(x1,(n_grid**2,))
    Xgrid[:,1] = np.reshape(x2,(n_grid**2,))
    
    blr_grid   = blr.predict_proba(Xgrid)[:,1]
    plt.figure(figsize=(8,6))
    plt.contourf(X1,X2,np.reshape(blr_grid,(n_grid,n_grid)),cmap="coolwarm")
    plt.plot(x[y==-1,0],x[y==-1,1],"bo", markersize = 3)
    plt.plot(x[y==1,0],x[y==1,1],"ro", markersize = 3)
    plt.colorbar()
    plt.title("Bayesian Logistic Regression, fitted with EM")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
#    
#    
#    # MUlticlass
#    from sklearn.datasets import make_blobs
#    centers = [(-3, 0), (0, 3), (3, 0)]
#    n_samples = 600
#    X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.0,
#                  centers=centers, shuffle=False, random_state=42)
#    blr        = BayesianLogisticRegression(solver = 'lbfgs_b')
#    blr.fit(X,y)
#    blr_grid   = blr.decision_function(X)
#    

    
    
    lr = LogisticRegression(C = 1e+10, fit_intercept = True)
    lr.fit(x,y)
    y_hat = lr.predict(x)
    plt.plot(x[y_hat==0,0],x[y_hat==0,1],'bo')
    plt.plot(x[y_hat==1,0],x[y_hat==1,1],'ro')
    print "Errors"
    print np.sum(blr.predict(x)!=y)
    print np.sum(y_hat!=y)
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(BayesianLogisticRegression)




       
if __name__ == "__main__":
    # generate features that will be highly correlated
    from sklearn.cross_validation import train_test_split 
    from sklearn.linear_model import LinearRegression
    np.random.seed(0)
    n, m              = 100, 30
    prototype         = 5*np.random.random(100)
    X                 = np.outer(prototype,np.ones(m)) + 0.05*np.random.randn(n, m)
    
    # Define only one feature to be relevant
    Theta             = np.zeros(m)
    Theta[0]          = 4.
    
    # Generate linear model with noise
    Y                 = np.dot(X, Theta) + np.random.normal(0,1,100)
    
    # train & test split
    X,x,Y,y           = train_test_split(X,Y, test_size = 0.4)
    
    # Fit the Bayesian Regression and an OLS for comparison
    br                = VariationalLinearRegression()
    br.fit(X,Y)
    
    ols               = LinearRegression()
    ols.fit(X, Y)
    
    plt.plot(ols.coef_,"r-",label = "OLS")
    plt.plot(br.coef_  ,"b-",label = "Bayesian")
    plt.plot(Theta,"g-", label = "Truth", linewidth = 3)
    plt.xlabel("parameter index")
    plt.ylabel("parameter estimarte")
    plt.title("OLS vs Bayesian Regression")
    plt.legend(loc = 1)
    plt.show()