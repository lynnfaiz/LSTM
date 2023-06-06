close all
clear all
clc
rng(1)  

%% load dataset
sourcePath = 'sourcepath';
load([sourcePath, 'datasetfile.mat'], 'inputs_trn', 'inputs_tst',...
    'outputs_trn', 'outputs_tst');

copydata = zeros(4,1); 
for station=1:1:1  
    station 
    XTrain90 = inputs_trn(:,:);
    YTrain90 = outputs_trn(station,:); 
    
    XTest10 = inputs_tst(:,:);
    YTest10 = outputs_tst(station,:);
    
    %select 1 output
    YTrain90 = YTrain90(:,:);
    YTest10 = YTest10(:,:) ;
    
    %Standardize Data
    mu = mean(XTrain90,2);
    sig = std(XTrain90,0,2);
    
    [rows,columns] = size(XTrain90);
    for i = 1:columns
        XTrain90(:,i) = (XTrain90(:,i) - mu) ./ sig;
    end
    
    [rows,columns] = size(XTest10);
    for i = 1:columns
        XTest10(:,i) = (XTest10(:,i) - mu) ./ sig;
    end
    
    % Define LSTM Network Architecture
    inputSize = 4;
    numResponses = 1; 
    numHiddenUnits = 4;
    
    layers = [ ...
        sequenceInputLayer(inputSize)
        lstmLayer(numHiddenUnits,'OutputMode','sequence')
        %bilstmLayer(numHiddenUnits,'OutputMode','sequence') % remove '%'
        %to run bilstm
        fullyConnectedLayer(10) 
        dropoutLayer(0.5)
        fullyConnectedLayer(numResponses)
        regressionLayer];
  
    opts = trainingOptions('adam', ...
        'MaxEpochs',300, ...
        'InitialLearnRate',0.01, ...
        'GradientThreshold',1, ...
        'Shuffle','never', ...
        'MiniBatchSize',128, ...
        'Verbose',0, ...
        'Plots','training-progress');
    
    %Train LSTM Network
    net = trainNetwork(XTrain90,YTrain90,layers,opts);
    %Predict Time Steps
    YPreds= predict(net,XTest10);
        
    % results
    rmse = sqrt(mean((YPreds-YTest10).^2));
    mse = rmse^2;
    mae = mae(YPreds-YTest10);
    mape = mean(abs((YPreds-YTest10)./YPreds));
        
    %copy this results   
    copydata(1,station)= rmse;
    copydata(2,station)= mse;
    copydata(3,station)= mae;
    copydata(4,station)= mape;
   
end


 