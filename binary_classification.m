clear all;clc;
%%load data
imds = imageDatastore('CNO_BINARY', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
% imds = imageDatastore('OM_BINARY', ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames');
% imds = imageDatastore('TR_BINARY', ...
%     'IncludeSubfolders',true, ...
%     'LabelSource','foldernames');

total_split=countEachLabel(imds)
num_images=length(imds.Labels);
%%
% Load Pretrained Network

% Number of folds
num_folds=3;

classifier_type=input('for EfficientNet type 1 and for ResNet type 2=');

% Loop for each fold
tic
for fold_idx=1:num_folds

    fprintf('Processing %d among %d folds \n',fold_idx,num_folds);

    % Test Indices for current fold
    test_idx=fold_idx:num_folds:num_images;

    % Test cases for current fold
    imdsTest = subset(imds,test_idx);

    % Train indices for current fold
    train_idx=setdiff(1:length(imds.Files),test_idx);

    % Train cases for current fold
    imdsTrain = subset(imds,train_idx);

    % Classifier Type
    if      classifier_type==1
        net = efficientnetb0
    elseif  classifier_type==2
        net = resnet50
    else
        print('classifier selection is wrong')
    end

    % analyzeNetwork(net)
    % net.Layers(1)
    inputSize = net.Layers(1).InputSize;

    %%
    % Replace Final Layers
    % lgraph = layerGraph(net.Layers);
    lgraph = layerGraph(net);


    [learnableLayer,classLayer] = findLayersToReplace(lgraph);

    numClasses = numel(categories(imdsTrain.Labels));

    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);

    lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
    newsoftmaxLayer = softmaxLayer('Name','new_softmax');

    if      classifier_type==1
        lgraph = replaceLayer(lgraph,'Softmax',newsoftmaxLayer);% Efficientb0
    elseif  classifier_type==2
        lgraph = replaceLayer(lgraph,'fc1000_softmax',newsoftmaxLayer); %Resnet50
    end


    classes = [ "CNO" "OTHER"];
    % classes = [ "OM" "OTHER"];
    % classes = [ "TR" "OTHER"];

    newClassLayer = classificationLayer('Name','new_classoutput');
    %     newClassLayer = classificationLayer('Name','new_classoutput','Classes',classes,'ClassWeights',classWeights);

    lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);


    %%
    % % Freeze Initial Layers
    % layers = lgraph.Layers;
    % connections = lgraph.Connections;
    %
    % layers(1:88) = freezeWeights(layers(1:88));
    % lgraph = createLgraphUsingConnections(layers,connections);

    %%
    %Data Augmentation
        pixelRange = [-15 15];
        scaleRange = [0.9 1.1];
        augmenter = imageDataAugmenter( ...
            'RandXReflection',true, ...
            'RandXTranslation',pixelRange, ...
            'RandYTranslation',pixelRange, ...
            'RandXScale',scaleRange, ...
            'RandYScale',scaleRange);

    miniBatchSize = 32;
  
    options = trainingOptions('adam',...
        'MaxEpochs',100,'MiniBatchSize',miniBatchSize,...
        'Shuffle','every-epoch', ...
        'InitialLearnRate',1e-4, ...
        'Verbose',false, ...
        'Plots','training-progress');

    augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
        'DataAugmentation',augmenter);
    % Train Network
    net = trainNetwork(augimdsTrain,lgraph,options);
    augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);


   
    %%
    %Classify Validation Images
    [predicted_labels(test_idx),posterior(test_idx,:)] = classify(net,augimdsTest);

    save(sprintf('ResNet50_%d_among_%d_folds',fold_idx,num_folds),'net','test_idx','train_idx');

    % Clearing unnecessary variables
    %     clearvars -except fold_idx num_folds num_images predicted_labels posterior imds netTransfer;
end

toc
% Actual Labels
actual_labels=imds.Labels;

% Confusion Matrix
figure;
plotconfusion(actual_labels,predicted_labels')
title('Confusion Matrix: Efficientb0');

test_labels=double(nominal(imds.Labels));
% figure
% ROC Curve - Our target class is the first class in this scenario
[fp_rate,tp_rate,T,AUC]=perfcurve(test_labels,posterior(:,2),2);
figure;
plot(fp_rate,tp_rate,'b-');
grid on;
xlabel('False Positive Rate');
ylabel('Detection Rate');
