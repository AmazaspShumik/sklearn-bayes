% created by Amazasp Shuamyan



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
        
        % maximum number of iterations and threshold for termination
        max_iter, thresh
        
        % vector of log - likelihoods ( with exclusion of constants)
        loglikes = ones(1);
        
    end
    
    
    methods(Access = 'public')
        
        function obj = BayesianRegression(x,y,evid_approx,max_iter)
            % Contructor
            % 
            % Parameters:
            %            x: matrix of size [n, m], matrix of explanatory variables
            %            y: vector of size [n, 1], vector of dependent variables
            %            evid_approx: str, method for evidence approximation,
            %                         can be only 'EM' or 'fixed-point' 
            %            max_iter: int, maximum number of iterations
            
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
            % Fits Bayesian Linear Regression with evidence approximation
            
            % find point estimates for alpha and beta using evidence approximation
            [alpha,beta]        = obj.evidenceApproximation();
            
            % Finds parameters of posterior distribution of weights
            [w_mu, w_precision] = obj.posteriorDistParams(alpha,beta)
        end
        
        function [prediction] = predict()
            
            
        end
    end
    
    methods(Access = 'private')
        
        function [alpha,beta] = evidenceApproximation(obj)
        % Calculates alpha ( precision parameter of prior distribution) and
        % beta ( precision parameter of likelihood) by maximising 
        % p(y|X,alpha,beta).
        %
        % Returns:
        %         alpha, beta - precison parameters ( both float)
        %
            
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
        
        
        function [mu,S] = posteriorDistParams(obj,alpha,beta)
        % Calculates parameters of posterior distribution of weights.
        % Since posterior distribution is Gaussian , it is characterised 
        % by mean and precision.
        % 
        % Parameters:
        %             alpha: float , precision of prior
        %             beta: float, precision of likelihood
        % 
        % Returns:
        %             mu: vector of size [obj.m x 1], posterior mean
        %             S:  matrix of size [obj.m x obj.m], posterior
        %                 precision
        %   
             dsq = diag(obj.D).^2;
             mu  = diag(obj.D)/(dsq + alpha/beta);
             S   = beta*obj.X'*obj.X + alpha;
        end
    end
    
end

