classdef BayesianRegression
    % Implements Bayesian Regression with evidence approximation ( type II
    % maximum likelihood) using fixed-point algorithm and EM.
    
    properties(GetAccess='public',SetAccess='public')
        
        % centered explanatory & dependent variables and their means
        Y, X, muX, muY
        
        % matrices of cvd decomposition
        U,D,V
        
        % precision parameters for likelihood and prior
        alpha, beta
        
        % input size parameters
        n,m
        
        % method for evidence approximation
        evid_approx_method
        
        % maximum number of iterations
        max_iter
        
        % vector of log - likelihoods ( with exclusion of constants)
        log
        
    end
    
    
    methods(Access = 'public')
        
        function obj = BayesianRegression(x,y,evid_approx,max_iter)
            % Contructor
            obj.evid_approx_method   = evid_approx;
            obj.max_iter             = max_iter;
            [obj.n,obj.m]            = size(x);
            obj.muX                  = mean(x);
            obj.muY                  = mean(y);
            obj.Y                    = y - obj.muY;
            obj.X                    = x - repmat(obj.muX,obj.n,1);
            [obj.U, obj.D, obj.V]    = svd(obj.X,'econ');
        end
        
        function fit(obj)
            % fits Bayesian Linear Regression
            
            % find point estimates for alpha and beta using evidence
            % approximation method
            [alpha,beta]        = obj.evidenceApproximation()
            
            % Using alpha and beta, find parameters of posterior
            % distribution of weights
            [w_mu, w_precision] = obj.posteriorDistParams()
        end
        
        function [prediction] = predict()
        end
    end
    
    methods(Access = 'private')
        
        function [alpha,beta] = evidenceApproximation(obj)
            
            % initialise alpha and beta
            alpha = rand(); beta = rand();
            
            % save squared diagonal ( no need to calculate this at each
            % iteration)
            dsq   = diag(obj.D).^2
            
            % Iterations of maximising algorithm
            for i = 1:obj.max_iter
                
                % find mean of posterior distribution (correposnds to
                % E-step in EM method for evidence approximation)                
                S     = diag(diag(obj.D)/(dsq + alpha/beta));
                mu    = obj.V' * S * obj.U' * obj.Y;
                
                % residuals
                error  = obj.Y - obj.X*mu;
                sqdErr = error'*error 
                
                if strcmp(obj.evid_approx_method,'fixed-point')
                    
                    % update gamma
                    gamma  = sum( beta*dsq/( beta * dsq + alpha) );
                    
                    % update alpha & beta
                    alpha  = gamma/(mu'*mu);
                    beta   = (N - gamma)/ sqdErr;
                    
                elseif strcomp(obj.evid_approx_method,'EM')
                    
                    % M-step, updates alpha and beta
                    alpha  = obj.m / ( mu'*mu + sum(1/(beta*dsq + alpha)));
                    beta   = obj.n / ( sqdErr + sum(dsq/(beta*dsq + alpha)));
                        
                end
                
                % calculates log-likelihood (excluding constants that do
                % not affect change in log-likelihood)
                norm      = obj.m/2*log(alpha) + obj.n/2*log(beta) - 1/2*sum(log(beta*dsq+alpha));
                loglike   = norm - alpha/2*(mu'*mu) - beta/2*sqdErr;
                log_likes = [log_likes,loglike];
                if i >= 2
                    if log
                        
                return [alpha,beta]
            end
            
            
            
        end
        
        
        function posteriorDistParams(obj)
            
            
        end
        
        
    end
    
end

