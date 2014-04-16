
%-------------------------------------------------------------------------
% Variational Inference on Latent Dirichlet Allocation (LDA)
% Function for computation on individual
% Author: Daniel Ribeiro Silva
% Email: drsilva@cs.cmu.edu
%-------------------------------------------------------------------------

function [Phi, Gamma, totalIterations] = IndividualVariationalInferenceLDA (ind, alpha, data, beta_matrix)


    %-------------------------------------
    % INITIALIZATION
    %-------------------------------------
    % Initialize variables
    %-------------------------------------
    
    N = size(beta_matrix,1);
    K = size(beta_matrix,2);

    idx = data(ind,:)>0;
    pos = find(data(ind,:) > 0);
    Ni = sum(idx);
    freq = data(ind,pos);
    alphaVector = alpha*ones(K,1);
    phiOld = (1/K)*ones(Ni, K); %from http://obphio.us/pdfs/lda_tutorial.pdf
    phiNew = zeros(Ni,K);
    gammaOld = alphaVector + N/K ; %from http://obphio.us/pdfs/lda_tutorial.pdf
    gammaNew = zeros(K,1);
    epsilon = 0.001; %convergence precision
    hasConverged = false;

    %-------------------------------------
    % VARIATIONAL INFERENCE
    %-------------------------------------
    % Run variational inference algorithm
    % Source1: Blei,Ng, and Jordan - LDA
    % Source2: http://obphio.us/pdfs/lda_tutorial.pdf
    %-------------------------------------

    t=0;
    while(~hasConverged)
        t = t + 1; 

        %update Phi
        for n = 1:Ni
            for k = 1:K
                phiNew(n, k) = beta_matrix(pos(n), k)*exp(psi(gammaOld(k)));  
            end
        end

        %normalize Phi using L1 norm
        phiNew = bsxfun(@rdivide,phiNew,sum(phiNew,2));

        %update gamma
        gammaNew = alphaVector + sum(bsxfun(@times,phiNew,freq'))';
        
        
        %check convergence
        if all(all(abs(phiNew-phiOld)<epsilon)) && all(all(abs(gammaNew-gammaOld)<epsilon))
            hasConverged = true;
        end

        phiOld = phiNew;
        gammaOld = gammaNew;

    end
    
    %-------------------------------------
    % RETURN
    %-------------------------------------
    % Return results
    %-------------------------------------

    Phi = phiNew;
    Gamma = gammaNew;
    totalIterations = t;

end




