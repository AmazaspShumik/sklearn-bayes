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
            
            % Iterations of maximising algorithm
            for i = 1:obj.max_iter
                
                % find mean of posterior distribution
                S     = diag(diag(obj.D)/(diag(obj.D).^2 + alpha/beta));
                mu    = obj.V' * S * obj.U' * obj.Y;
                % residuals
                error = obj.Y - obj.X*mu
                
                if strcmp(obj.evid_approx_method,'fixed-point')
                    
                    % update gamma
                    
                elseif strcomp(obj.evid_approx_method,'EM')
                    
                        
                end
                
                
            end
            
            
            
        end
        
        
        function posteriorDistParams(obj)
            
            
        end
        
        
    end
    
end

