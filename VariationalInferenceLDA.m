
%-------------------------------------------------------------------------
% Variational Inference on Latent Dirichlet Allocation (LDA)
% Script for computation on all individuals
% Author: Daniel Ribeiro Silva
% Email: drsilva@cs.cmu.edu
%-------------------------------------------------------------------------

clearvars;clc;close all;

%-------------------------------------
% LOAD DATA
%-------------------------------------
% Load data that contains genotype
% information form individuals
%-------------------------------------

load('data.mat');

M = size(data,1);
N = size(beta_matrix,1);
K = size(beta_matrix,2);

%-------------------------------------
% INITIALIZATION
%-------------------------------------
% Initialize variables
%-------------------------------------

ind = 1; %index of individual
alpha = 0.1;
alphaExperiments = [0.01 0.1 1 10];
Theta = zeros(M,K); %gamma for each individual
totalIterations = 0;

%-------------------------------------
% GENOTYPE LOCI INFERENCE
%-------------------------------------
% LDA inference of phi for each 
% genotype locus present in a given 
% individual (ind)
%-------------------------------------

[phiInd, gammaInd, tIt] = IndividualVariationalInferenceLDA (ind, alpha, data, beta_matrix);

figure(1);
imagesc(phiInd);colorbar;
title(strcat('PHI  for  alpha = ', num2str(alpha)));

%-------------------------------------
% ANCESTOR ASSIGNMENT
%-------------------------------------
% LDA inference of gamma for each
% individual in the population
%-------------------------------------

for experiment = 1:size(alphaExperiments,2)
    tic;
    currentAlpha = alphaExperiments(experiment);
    totalIterations = 0;
    
    for m = 1:M
        [phiInd, gammaInd, tIt] = IndividualVariationalInferenceLDA (m, currentAlpha, data, beta_matrix);
        totalIterations = totalIterations + tIt;
        Theta(m,:) = gammaInd;
    end
    
    fprintf('alpha = %f\n', currentAlpha);
    fprintf('total iterations = %i\n', totalIterations);
    fprintf('run time = %.2f sec.\n', toc);
    fprintf('\n-------------------------\n');
    
    figure(1+experiment);
    imagesc(Theta);colorbar;
    title(strcat('THETA  for  alpha = ', num2str(currentAlpha)));
    
end

