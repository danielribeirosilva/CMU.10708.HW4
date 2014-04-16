
%-------------------------------------------------------------------------
% Gibbs Sampling for Inference of Mixture of Gaussians
% Author: Daniel Ribeiro Silva
% Email: drsilva@cs.cmu.edu
%-------------------------------------------------------------------------

%------------------------------------------------------------------------+
% PGM of the model                                                       |
%                                                                        |
%                  _____________        ___________________________      |
%     ______       |  _______  |        |   _____           _____  |     |
%    /      \      | /       \ |        |  /     \         /     \ |     |
%    | \tau | ---->| | \mu_k | | -------+->| x_i | <------ | z_i | |     |
%    \______/      | \_______/ |        |  \_____/         \_____/ |     |
%                  |           |        |                          |     |
%                  |     k=1..K|        |                   i=1..N |     |
%                  +-----------+        +--------------------------+     |
%                                                                        |
%------------------------------------------------------------------------+

clearvars;clc;close all;

%-------------------------------------
% GENERATE DATA
%-------------------------------------
% Forward sample 100 observations
%-------------------------------------

totalComponents = 2;
muTrue = [-5 5];
mixtureWeights = [0.5 0.5];
tau = 10;
totalObservations = 100;
componentObserved = randsample(1:totalComponents,totalObservations,true,mixtureWeights);

X = zeros(1,totalObservations);
for i = 1:totalObservations
    component = componentObserved(i);
    muObs = muTrue(component);
    X(i) = normrnd(muObs,sqrt(tau));
end

%-------------------------------------
% RUN SAMPLING
%-------------------------------------
% Sample using Gibbs Sampling
%-------------------------------------

totalSamples = 2000;

%initialize variables
Z = zeros(totalObservations, totalSamples+1);
Mu = zeros(totalComponents, totalSamples+1);
N = zeros(totalComponents, totalSamples+1);
Mu(:,1) = [-1 1]; %can be any random values
sumXik = zeros(totalComponents,1);

for t=2:totalSamples+1
    
    %sample z_i ~ Categorical(W) with w_k \propto exp( - (x_i - mu_k)^2 )
    for i = 1:totalObservations
        weights = exp( - (X(i) - Mu(:,t-1)).^2 );
        Z(i,t) = randsample(1:totalComponents,1,true,weights);
    end
    
    %compute N_k and sum x_ik
    for k = 1:totalComponents
        idx = (Z(:,t) == k);
        N(k,t) = sum( idx );
        sumXik(k) = sum( X( idx') );
    end
    
    %sample mu_k ~ Normal( tau*(sum x_ik)/(N_k*tau + 1), tau/(N_k*tau + 1)) 
    for k = 1:totalComponents
        current_mu = tau * sumXik(k) / ( N(k,t)*tau + 1 );
        current_sigma = sqrt( tau / ( N(k,t)*tau + 1 ) );
        Mu(k,t) = normrnd(current_mu, current_sigma);
    end
    
end


%-------------------------------------
% DISCARD INITIAL SAMPLES
%-------------------------------------
% Discard first samples as burn-in
%-------------------------------------

totalDiscarded = 500;

ZFinal = Z(:,totalDiscarded+2:end);
MuFinal = Mu(:,totalDiscarded+2:end);


%-------------------------------------
% RESULTS
%-------------------------------------
% Plot and analyze results
%-------------------------------------

for k=1:totalComponents
    
    %plot estimated curve for the k-th component
    figure(k);
    y = MuFinal(k,:);
    delta = (max(y)-min(y))/30;
    x = min(y):delta:max(y);
    [counts binValues] = hist(y,x);
    hist(y,x);
    title(strcat('component ',num2str(k)));
    
    %plot true distribution line on top
    deltaTrue = (max(y)-min(y))/1000;
    xTrue = min(y):deltaTrue:max(y);
    yTrue = (1/sqrt(2*pi*tau))*exp(-(xTrue - muTrue(k)).^2 / 2*tau);
    yTrue = yTrue * max(counts) / max(yTrue); %adjust curve to plotted histogram
    hold on;
    plot(xTrue,yTrue,'r');
end


