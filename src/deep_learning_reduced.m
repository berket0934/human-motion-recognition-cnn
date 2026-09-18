%% 0. Setup and Constants 
WINDOW_LENGTH = 500;        % Time steps per segment (Fixed sequence length)
WINDOW_OVERLAP = 250;       % 50% overlap
inputSize = 6;              % Number of input features
ignoredLabels = [3,4,5,9,10,11,12];   % Labels to ignore

train_path = "Training\*.csv";
validation_path = "Validation\*.csv";
test_path = "Testing\*.csv";

read_size = 1.5e8;

%% Create datastores
ds = tabularTextDatastore(train_path, 'ReadSize', read_size);
dsValidation = tabularTextDatastore(validation_path, 'ReadSize', read_size);
dsTest = tabularTextDatastore(test_path, 'ReadSize', read_size);

%% Preprocess Function
function out = preprocessData(T, WINDOW_LENGTH, WINDOW_OVERLAP, ignoredLabels)
    % Extract and Prepare Data
    numCols = size(T, 2);
    Xtable = T(:, 2:numCols-1);  % Features (excluding time and final label)
    Ytable = T(:, end);          % Labels
    
    X = varfun(@double, Xtable);
    X = table2array(X);
    Y_all_labels = table2array(Ytable);
    
    % Normalize features per sequence
    mu = mean(X, 1);
    sigma = std(X, 0, 1);
    epsilon = 1e-6;
    sigma(sigma == 0) = 1;
    X_norm = (X - mu) ./ (sigma + epsilon);

    % Sliding window logic
    numSteps = size(X_norm, 1);
    startIdx = 1:(WINDOW_LENGTH - WINDOW_OVERLAP):(numSteps - WINDOW_LENGTH + 1);

    X_windows = {};
    Y_labels = {};

    for i = 1:length(startIdx)
        s = startIdx(i);
        e = s + WINDOW_LENGTH - 1;
        X_window = X_norm(s:e, :);
        Y_window_labels = Y_all_labels(s:e);
        Y_mode = mode(Y_window_labels);

        % Ignored labels
        if ismember(Y_mode, ignoredLabels)
            continue;
        end

        % Transpose X_window: [500x6] -> [6x500x1]
        X_windows{end+1} = reshape(X_window', [size(X_window, 2), WINDOW_LENGTH, 1]);

        Y_labels{end+1} = categorical(Y_mode);
    end

    % Return as table
    out = table(X_windows', Y_labels', 'VariableNames', {'X', 'Y'});
end

%% Apply transform to training, validation and test
dsTransformed = transform(ds, @(T) preprocessData(T, WINDOW_LENGTH, WINDOW_OVERLAP, ignoredLabels));
dsValidationTransformed = transform(dsValidation, @(T) preprocessData(T, WINDOW_LENGTH, WINDOW_OVERLAP, ignoredLabels));
dsTestTransformed = transform(dsTest, @(T) preprocessData(T, WINDOW_LENGTH, WINDOW_OVERLAP, ignoredLabels));

%% Remove ignored labels from the datastores
% Read all transformed data into tables
dataTrain = readall(dsTransformed);       % table with columns X (cell) and Y (cell of categorical)
dataVal   = readall(dsValidationTransformed);
dataTest  = readall(dsTestTransformed);

% Convert cell-of-categoricals into a single categorical vector for each set
if ~isempty(dataTrain)
    trueTrainVec = vertcat(dataTrain.Y{:});    % categorical vector
    trueTrainVec = removecats(trueTrainVec);   % remove unused categories
    remainingCats = categories(trueTrainVec);
    numClasses = numel(remainingCats);

    % Rebuild the table's Y column as a cell array of categorical scalars with the cleaned categories
    dataTrain.Y = arrayfun(@(k) categorical(trueTrainVec(k), remainingCats), ...
                           (1:numel(trueTrainVec))', 'UniformOutput', false);
else
    error('No training samples after preprocessing. Check your ignoredLabels or preprocessing.');
end

% Validation set
if ~isempty(dataVal)
    trueValVec = vertcat(dataVal.Y{:});
    trueValVec = categorical(trueValVec, remainingCats); % align categories to training set
    trueValVec = removecats(trueValVec);
    dataVal.Y = arrayfun(@(k) categorical(trueValVec(k), remainingCats), ...
                         (1:numel(trueValVec))', 'UniformOutput', false);
end

% Test set
if ~isempty(dataTest)
    trueTestVec = vertcat(dataTest.Y{:});
    trueTestVec = categorical(trueTestVec, remainingCats); % align categories to training set
    trueTestVec = removecats(trueTestVec);
    dataTest.Y = arrayfun(@(k) categorical(trueTestVec(k), remainingCats), ...
                          (1:numel(trueTestVec))', 'UniformOutput', false);
end

% Display remaining classes
disp("Remaining Classes:");
disp(remainingCats);
disp(['numClasses = ', num2str(numClasses)]);

% Convert back to datastores
dsTransformed = arrayDatastore(dataTrain, 'OutputType', 'same');
dsValidationTransformed = arrayDatastore(dataVal, 'OutputType', 'same');
dsTestTransformed = arrayDatastore(dataTest, 'OutputType', 'same');

%% Network layers
layers = [
    sequenceInputLayer(inputSize, 'MinLength', WINDOW_LENGTH)
    
    convolution1dLayer(10, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2, 'Stride', 2)
    
    convolution1dLayer(10, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2, 'Stride', 2)
    
    convolution1dLayer(10, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling1dLayer(2, 'Stride', 2)
    
    dropoutLayer(0.4)
    globalAveragePooling1dLayer
    
    fullyConnectedLayer(numClasses)
    softmaxLayer
];

%% Training options
options = trainingOptions('adam', ...
    'MaxEpochs', 75, ...
    'MiniBatchSize', 16, ...
    'ValidationData', dsValidationTransformed, ...
    'Metrics', ["accuracy","fscore","recall"], ...
    'ObjectiveMetricName', 'fscore', ...
    'OutputNetwork', 'best-validation', ...
    'InitialLearnRate', 1e-4, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 25, ...
    'Shuffle', 'never', ...
    'ExecutionEnvironment', 'gpu', ...
    'Verbose', 1, ...
    'Plots', 'training-progress', ...
    'InputDataFormats', {'CTB'});

%% Train network
[net, info] = trainnet(dsTransformed, layers, 'crossentropy', options);

%% Test and evaluation
% Prepare XTest (transpose back to cell of [T x F] sequences for minibatchpredict)
dataTest = readall(dsTestTransformed);   % table with X (cell) and Y (cell)
XTest = dataTest.X;
XTestCorrected = cellfun(@(x) x', XTest, 'UniformOutput', false);

trueLabelsCat = vertcat(dataTest.Y{:});

% Prediction
disp('Starting minibatchpredict...');
YPredScores = minibatchpredict(net, XTestCorrected, ...
    'MiniBatchSize', 16, ...
    'ExecutionEnvironment', 'cpu');

% Convert scores to labels
categoryList = remainingCats;
predictedLabels = scores2label(YPredScores, categoryList);

% Accuracy
isCorrect = (predictedLabels == trueLabelsCat);
accuracy = sum(isCorrect) / numel(isCorrect);

disp('--------------------------------------------------');
disp(['Total Test Sequences: ', num2str(numel(trueLabelsCat))]);
disp(['Predicted Labels Count: ', num2str(numel(predictedLabels))]);
disp(['Correctly Classified: ', num2str(sum(isCorrect))]);
disp(['Final Test Accuracy: ', num2str(accuracy * 100), '%']);
disp('--------------------------------------------------');

figure

oldNames = ["1", "2", "6", "7", "8"];
newNames = ["walking", "running", "standing", "sitting", "lying"];

% Rename the categories
trueLabelsCat = renamecats(trueLabelsCat, oldNames, newNames);
predictedLabels = renamecats(predictedLabels, oldNames, newNames);

confusionchart(trueLabelsCat, predictedLabels);
title('CNN Activity Classification Confusion Matrix');


