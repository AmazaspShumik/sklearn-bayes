% Small demo for bayesian regression

% generate data 
X       = rand(100,100);
X(:,1)  = linspace(-5,5); 
Y       = 10*X(:,1) + randn(100,1);

% fit bayesian regression with EM fitting method
br = BayesianRegression(X,Y,'EM',100,1e-3);
br.fit();

% predict mean and variance of predictive distribution
[y_hat,y_var] = br.predict_dist(X);

% plot data
plot(X(:,1),Y,'b-','linewidth',3)
hold on
plot(X(:,1),y_hat,'ro', 'markersize',3)
title('Bayesian Regression')
xlabel('x')
ylabel('y')
hold off
