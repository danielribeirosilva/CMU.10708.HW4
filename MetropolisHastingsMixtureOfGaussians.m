
%-------------------------------------------------------------------------
% Metropolis-Hastings for Inference of Mixture of Gaussians
% Author: Daniel Ribeiro Silva
% Email: drsilva@cs.cmu.edu
%-------------------------------------------------------------------------

%------------------------------------------------------------------------+
% PGM of the model                                                       |
%                                                                        |
%                  _____________        _____________                    |
%     ______       |  _______  |        |   _____   |                    |
%    /      \      | /       \ |        |  /     \  |                    |
%    | \tau | ---->| | \mu_k | | -------+->| x_i |  |                    |
%    \______/      | \_______/ |        |  \_____/  |                    |
%                  |           |        |           |                    |
%                  |     k=1..K|        |    i=1..N |                    |
%                  +-----------+        +-----------+                    |
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
% SAMPLING + REJECTING
%-------------------------------------
% Sample using Metropolis-Hasting
%-------------------------------------

totalSamples = 2000;

%initialize variables
Mu = zeros(totalComponents, totalSamples+1);
Mu(:,1) = [-1 1]; %can be any random values
%likelihood of first Mu
Xrep = repmat(X,totalComponents,1);
logPX = sum(log(sum(bsxfun(@times,exp(-bsxfun(@minus,Xrep,Mu(:,1)).^2),mixtureWeights')))); %log to avoid underflow
logPPrior = sum(log(exp(-(Mu(:,1).^2)/(2*tau))));
logPPosteriorOld = logPX + logPPrior;

t=2;
while t <= totalSamples+1
    
    %sample from proposal distribution mu ~ Gaussian(0,20I)
    MuProposal = normrnd(0,sqrt(20),totalComponents,1);
    
    %compute data likelihood
    logPX = sum(log(sum(bsxfun(@times,exp(-bsxfun(@minus,Xrep,MuProposal).^2),mixtureWeights')))); %log to avoid underflow
    logPPrior = sum(log(exp(-(MuProposal.^2)/(2*tau))));
    logPPosteriorNew = logPX + logPPrior;
    
    %compute accpetance probability
    A = min([1, exp(logPPosteriorNew-logPPosteriorOld)]);
    
    %check if accepts
    if rand() <= A
       Mu(:,t) =  MuProposal;
       t = t + 1;
       logPPosteriorOld = logPPosteriorNew;
       disp(strcat('accepted:',num2str(t)));
    end
    
end


%-------------------------------------
% DISCARD INITIAL SAMPLES
%-------------------------------------
% Discard first samples as burn-in
%-------------------------------------

totalDiscarded = 500;

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
    delta = (max(y)-min(y))/100;
    x = (min(y)-1):deltaTrue:(max(y)+1);
    [counts binValues] = hist(y,x);
    hist(y,x);
    title(strcat('component ',num2str(k)));
    
    %plot true distribution line on top
    deltaTrue = (max(y)-min(y))/1000;
    xTrue = (min(y)-1):deltaTrue:(max(y)+1);
    yTrue = sum(exp(-(bsxfun(@minus,repmat(xTrue,totalComponents,1),muTrue(1:totalComponents)')).^2 / 2*tau));
    yTrue = yTrue * max(counts) / max(yTrue); %adjust curve to plotted histogram
    hold on;
    plot(xTrue,yTrue,'r');
end


