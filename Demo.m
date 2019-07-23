% run this demo.
clear
clc
warning off;

name = 'PaviaU';
load(sprintf('dataset_%s.mat', name));

param.flag = 0; param.tol = 10^-4; param.maxIter = 10^2;
% set parameters related to the SOPSRL model
param.tau1 = 0.01; param.tau2 = 10^-5; param.tau3 = 10^5;
param.k = 10; param.r = 2;

B = size(data,1); % band dimension
n = length(trainIdx); % number of samples
%% main program
flag_S = zeros(n, 1); 
tauMin_S = 50/n; tauMax_S = 300/n; % decide the number of samples at time t.
tauMin_B = 1/3; tauMax_B = 2/3; % decide the number of available bands at time t.
totalTime = 0; tCount = 0;
xtrain = data(:, trainIdx);
q = zeros(B,1);
while true
    if sum(flag_S) == n
        % deciding whether to end
        break;
    end
    %% simulate the available samples with some bands
    tau_B = tauMin_B + (tauMax_B - tauMin_B)*rand(1);  % deciding the number of bands of subsamples
    bandSet = zeros(floor(tau_B*B), 1); % mark the available bands at current time
    count2 = 0;
    while sum(bandSet == 0) ~= 0
        j = floor(1+B*rand(1));
        if sum(bandSet == j) == 0; count2 = count2 + 1; bandSet(count2) = j; end
    end
    tau_S = tauMin_S + (tauMax_S - tauMin_S)*rand(1);
    if sum(flag_S==0) <= 300
        sampleSet = find(flag_S == 0);
        flag_S(flag_S == 0) = 1;
    else
        sampleSet = zeros(floor(tau_S*n), 1); % mark the available samples at current time
        count2 = 0;
        while sum(sampleSet == 0) ~= 0
            j = floor(1+n*rand(1));
            if flag_S(j) == 0 && sum(sampleSet == j) == 0 && sum(xtrain(bandSet,j)) ~= 0
                count2 = count2 + 1; 
                sampleSet(count2) = j; 
            end
        end
        flag_S(sampleSet) = 1;
    end         
    %% model learning
    bandSet = sort(bandSet);
    X = xtrain(bandSet, sampleSet);
    if length(sampleSet) <= param.k
        param.k = length(sampleSet)-1;
    end
    W = zeros(length(bandSet), length(bandSet));
    S = zeros(length(sampleSet), length(sampleSet));
    % init similarity matrix S according to the correlation
    for i = 1:length(sampleSet)
        tmpS = X(:,i)' * X;
        tmpS = tmpS';
        [SY, SI] = sort(tmpS, 'descend');
        S(SI(1:param.k), i) = SY(1:param.k) ./ sum(SY(1:param.k));
    end
    phi = 1/length(sampleSet) .* ones(length(sampleSet),1);
    tstart = tic;
    [W, phi, S] = SOPSRL(X, S, W, phi, q(bandSet), param); % main function
    telasped = toc(tstart);
    totalTime = totalTime + telasped;
    tCount = tCount + 1;
    if sum(sum(isnan(W))) == 0
        q(bandSet) = (sum(W.^2, 2)).^(1/2);
    end
end
fprintf('training model ends. Total time: %d(s), Single time: %d(s).\n', totalTime, totalTime/tCount);
%% select expected bands
[qY, qI] = sort(q, 'descend');
bandNum = 30;
selectedBands = sort(qI(1:bandNum));
%% training classifier and perform classification
xtrain2 = xtrain(selectedBands,:); ytrain = labels(trainIdx);
xtest2 = data(selectedBands, testIdx); ytest = labels(testIdx);
optimalNumNeighbors = 0;
optimalLoss = 10^2;
kArray = [2, 3, 5, 7, 10, 15];  % the value of K for KNN
for kId = 1:length(kArray)
    mdl = ClassificationKNN.fit(xtrain2', ytrain, 'NumNeighbors', kArray(kId));
    cvmdl = crossval(mdl, 'kfold', 5);
    cvmdlloss = kfoldLoss(cvmdl);
    if cvmdlloss < optimalLoss
        optimalLoss = cvmdlloss;
        optimalNumNeighbors = kArray(kId);
    end
end
mdl = ClassificationKNN.fit(xtrain2', ytrain, 'NumNeighbors', optimalNumNeighbors);
tlabs = predict(mdl, xtest2');
[kappa, acc, acc_O, acc_A] = evaluate_results(tlabs, ytest);
fprintf('numbef of bands: %d, kappa: %d, OA: %d, AA: %d.\n', bandNum, kappa, acc_O, acc_A);