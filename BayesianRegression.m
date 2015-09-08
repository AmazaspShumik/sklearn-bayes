


classdef BayesianRegression < handle
    % Implements Bayesian Regression with evidence approximation ( type II
    % maximum likelihood) using fixed-point algorithm and EM.
    
    properties(GetAccess='public',SetAccess='public')
        
        % centered explanatory & dependent variables and their means
        Y, X, muX, muY
        
        % matrices of cvd decomposition
        U,D,V
        
        % precision parameters for likelihood and prior
        alpha, beta
        
        % posterior mean and covariance for weights
        mu, S
        
        % input size parameters
        n,m
        
        % method for evidence approximation
        evid_approx_method
        
        % maximum number of iterations and threshold for termination
        max_iter, thresh
        
        % vector of log - likelihoods ( with exclusion of constants)
        loglikes = ones(1);
        
    end
    
    
    methods(Access = 'public')
        
        function obj = BayesianRegression(x,y,evid_approx,max_iter, thresh)
            % Contructor
            % 
            % Parameters:
            % ----------
            %            x: matrix of size [n, m], matrix of explanatory variables
            %            y: vector of size [n, 1], vector of dependent variables
            %            evid_approx_method: str, method for evidence approximation,
            %                                can be only 'EM' or 'fixed-point' 
            %            max_iter: int, maximum number of iterations
            
            obj.evid_approx_method   = evid_approx;
            obj.thresh               = thresh;
            obj.max_iter             = max_iter;
            [obj.n,obj.m]            = size(x);
            obj.muX                  = mean(x);
            obj.muY                  = mean(y);
            obj.Y                    = bsxfun(@minus,y,obj.muY);
            obj.X                    = bsxfun(@minus,x,obj.muX);
            [obj.U, obj.D, obj.V]    = svd(obj.X,'econ');
        end
        
        
        function obj = fit(obj)
            % Fits Bayesian Linear Regression with evidence approximation
            
            % find point estimates for alpha, beta and mu by maximising
            % marginal likelihood
            [obj.alpha, obj.beta, obj.mu]    = obj.evidenceApproximation();
            
            % Find covariance
            d                      = 1 /( obj.beta*diag(obj.D).^2 + obj.alpha);
            obj.S                  = obj.V'*diag(d)*obj.V;
            
        end
        
        
        function [t,tVar] = predict_dist(obj,X)
            % Calculates mean and variance of predictive distribution
            % 
            % Parameters:
            % -----------
            %            X: matrix of size [unknown, m], matrix of explanatory 
            %               variables
            %
            % Returns:
            % --------
            %            t: vector of size [unknown, 1], mean of predictive
            %               distribution
            %            tVar: vector of size [unknown, 1], variance of
            %                  predictive distribution
            %
            x    = X - obj.muX;
            t    = x*obj.mu + obj.muY;
            tVar = 1/obj.beta + x'*obj.S*x;
        end
        
        
        function [t] = predict(obj,X)
            % Calculates mean of predictive distribution
            % 
            % Parameters:
            % -----------
            %            X: matrix of size [unknown, m], matrix of explanatory 
            %               variables
            %
            % Returns:
            % --------
            %            t: vector of size [unknown, 1], mean of predictive
            %               distribution
            %
            x = X - obj.muX;
            t = x*obj.mu + obj.muY;
        end
        
    
        
        function [alpha,beta,mu] = evidenceApproximation(obj)
            % Calculates alpha (precision parameter of prior distribution),
            % beta ( precision parameter of likelihood) and mu ( mean of posterior
            % distribution) by maximising p(y|X,alpha,beta).
            %
            % Returns:
            % --------
            %         alpha, beta, mu - precision parameters and posterior
            %                           mean
            %
            
            % initialise alpha and beta
            alpha = rand(); beta = rand();
            
            % save squared diagonal ( no need to calculate this at each
            % iteration)
            dsq       = diag(obj.D).^2;
            
            % Iterations of maximising algorithm
            for i = 1:obj.max_iter
                
                % find mean of posterior distribution (correposnds to
                % E-step in EM method for evidence approximation)                
                S     = diag(obj.D)/(dsq + alpha/beta);
                mu    = obj.V' * S * obj.U' * obj.Y;
                
                % residuals
                error  = obj.Y - obj.X*mu;
                sqdErr = error'*error;
                
                if strcmp(obj.evid_approx_method,'fixed-point')
                    
                    % update gamma
                    gamma  = sum( beta * dsq / (beta * dsq + alpha) );
                    
                    % update alpha & beta
                    alpha  = gamma/(mu'*mu);
                    beta   = (N - gamma)/ sqdErr;
                    
                elseif strcmp(obj.evid_approx_method,'EM')
                    
                    % M-step, updates alpha and beta
                    alpha  = obj.m / ( mu'*mu + sum(1/(beta*dsq + alpha)));
                    beta   = obj.n / ( sqdErr + sum(dsq/(beta*dsq + alpha)));
                        
                end
                
                % after alpha & beta are updated last time we should also
                % update mu
                S     = diag(obj.D)/(dsq + alpha/beta);
                mu    = obj.V' * S * obj.U' * obj.Y;
                
                % calculates log-likelihood (excluding constants that do
                % not affect change in log-likelihood)
                norm         = obj.m/2*log(alpha) + obj.n/2*log(beta) - 1/2*sum(log(beta*dsq+alpha));
                loglike      = norm - alpha/2*(mu'*mu) - beta/2*sqdErr;
                obj.loglikes = [obj.loglikes,loglike];
                
                % terminate iterations if change in log -likelihood is
                % smaller than threshold
                if i >= 2
                    if obj.loglikes(i) - obj.loglikes(i-1) < obj.thresh
                        return 
                    end
                end
            end 
        end
    end
end

