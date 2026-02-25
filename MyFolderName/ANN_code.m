NewFolderName/ANN_Model.m

clear; clc;

inputdata1 = readmatrix('addExperiment.csv', 'Range', 'A2:F132'); % Input features
outputdata1 = readmatrix('addExperiment.csv', 'Range', 'G2:G132'); % Target values

% Normalize input features
inputdata_mean = mean(inputdata1, 1);
inputdata_std = std(inputdata1, 0, 1);
inputdata = (inputdata1 - inputdata_mean) ./ inputdata_std;

% Normalize output (target values)
outputdata_mean = mean(outputdata1, 1);
outputdata_std = std(outputdata1, 0, 1);
outputdata = (outputdata1 - outputdata_mean) ./ outputdata_std;

% Separate data for training and testing
rng('default');
num_samples = size(inputdata, 1); 
random = sort(randsample(num_samples, 10)); 

% Separate testing data
input = inputdata(random, :); 
data_output = outputdata(random, :); 

% Separate training data
notselected = setdiff(1:num_samples, random); 
ann_input = inputdata(notselected, :); 
ann_output = outputdata(notselected, :); 

disp('Data for ANN and data left for final testing are divided');

%% Train data with single layer ANN 
clear x y numfolds c h net tr Hmin Hmax
clear Testperf_record Trainperf_record train_idx test_idx;
clear xTrain yTrain xTest yTest ypredict ytrain_net ycom ycomtr
clear Testperf Trainperf testRcal testR testR2 trainRcal trainR trainR2
clear stopcrit bestepoch runtime

x = ann_input; % All input data
y = ann_output; % All output data
numfolds = 3; % Number of folds
c = cvpartition(size(y, 1), 'KFold', numfolds); % Partition data for k-fold cross-validation

%table to store results
Testperf_record = zeros(numfolds,1);
Trainperf_record = zeros(numfolds,1);

Hmin = 1; % Minimum hidden nodes
Hmax = 60; % Adjust for faster training
h2 = 25;

for h = Hmin:Hmax
    net = fitnet([h, h2], 'trainlm');

    
    for i = 1:numfolds
        rng(1);
        % Divide data for train and test for each fold
        fieldfold = sprintf('Fold%d', i);
        fieldfold1 = sprintf('Node%d', h);
        train_idx.(fieldfold1)(:, i) = c.training(i); %return logical index for training in each of fold
        test_idx.(fieldfold1)(:, i) = c.test(i); %return logical index for testing in each of fold
        xTrain.(fieldfold1).(fieldfold) = x(train_idx.(fieldfold1)(:, i), :);
        yTrain.(fieldfold1).(fieldfold) = y(train_idx.(fieldfold1)(:, i), :);
        xTest.(fieldfold1).(fieldfold) = x(test_idx.(fieldfold1)(:, i), :);
        yTest.(fieldfold1).(fieldfold) = y(test_idx.(fieldfold1)(:, i), :);

        % Set Division function
        net.divideFcn = 'dividerand'; % Random division
        net.trainParam.showWindow = false;
        % Setup Division of Data for Training, Validation, Testing
        net.divideParam.trainRatio = 80 / 100;
        net.divideParam.valRatio = 10 / 100;
        net.divideParam.testRatio = 10 / 100;

        net.performParam.normalization = 'standard';

        % Set function of hidden layers
        net.layers{1}.transferFcn = 'radbasn';
        net.layers{2}.transferFcn = 'radbasn';
        % Train the model
        [netre, tr] = train(net, [xTrain.(fieldfold1).(fieldfold)]', [yTrain.(fieldfold1).(fieldfold)]');
        ypredict = netre([xTest.(fieldfold1).(fieldfold)]');
        ytrain_net = netre([xTrain.(fieldfold1).(fieldfold)]');
        stopcrit{i, h} = tr.stop;
        bestepoch(i, h) = tr.best_epoch;
        runtime(i, h) = tr.time(end);
        Testperf = perform(netre, yTest.(fieldfold1).(fieldfold), ypredict);
        Testperf_record(i, h) = Testperf; % Record value of MSE (Loss calculation of test)
        Trainperf = perform(netre, yTrain.(fieldfold1).(fieldfold), ytrain_net);
        Trainperf_record(i, h) = Trainperf; % Record value of MSE (Loss calculation of train)
        testRcal = corrcoef(ypredict, [yTest.(fieldfold1).(fieldfold)]');
        testR(i, h) = testRcal(1, 2);
        testR2(i, h) = (testR(i, h))^2;
        trainRcal = corrcoef(ytrain_net, [yTrain.(fieldfold1).(fieldfold)]');
        trainR(i, h) = trainRcal(1, 2);
        trainR2(i, h) = (trainR(i, h))^2;
        testRMSE(i, h) = sqrt(mean((ypredict - yTest.(fieldfold1).(fieldfold)').^2)); %normalized RMSE
        trainRMSE(i, h) = sqrt(mean((netre(xTrain.(fieldfold1).(fieldfold)') - yTrain.(fieldfold1).(fieldfold)').^2)); %normalized RMSE
        % testRMSE_raw(i, h) = sqrt(mean((ypredict - yTest.(fieldfold1).(fieldfold)').^2)); %raw RMSE
        % trainRMSE_raw(i, h) = sqrt(mean((netre(xTrain.(fieldfold1).(fieldfold)') - yTrain.(fieldfold1).(fieldfold)').^2)); %raw RMSE
        testMAE(i, h) = mean(abs(ypredict - yTest.(fieldfold1).(fieldfold)')); %normalized MAE
        trainMAE(i, h) = mean(abs(netre(xTrain.(fieldfold1).(fieldfold)') - yTrain.(fieldfold1).(fieldfold)')); %normalized MAE
   
        netcom.(fieldfold1).(fieldfold) = netre;
        ycom.(fieldfold1).(fieldfold) = ypredict;
        ycomtr.(fieldfold1).(fieldfold) = ytrain_net;
        trcom.(fieldfold1).(fieldfold) = tr;

      

        net = init(net);
    end
end


% extract value of ycom
% Initialize storage for selected models
selected_models = struct();

% Define the specific cases to save
cases_to_save = [
    struct('h', 17, 'i', 2);
    struct('h', 18, 'i', 1);
    struct('h', 19, 'i', 1);
    struct('h', 46, 'i', 2);
    struct('h', 49, 'i', 2);
];


% Loop through the specified cases and save models
for idx = 1:numel(cases_to_save)
    h = cases_to_save(idx).h; % Get the number of hidden nodes
    i = cases_to_save(idx).i; % Get the specific fold index

    % Generate field names for accessing the stored results
    fieldfold1 = sprintf('Node%d', h);
    fieldfold = sprintf('Fold%d', i);

    % Store the selected models and their results
    selected_models.(fieldfold1).(fieldfold).net = netcom.(fieldfold1).(fieldfold); % Save trained network
    selected_models.(fieldfold1).(fieldfold).tr = trcom.(fieldfold1).(fieldfold); % Save training record
    selected_models.(fieldfold1).(fieldfold).ypredict = ycom.(fieldfold1).(fieldfold); % Save test predictions
    selected_models.(fieldfold1).(fieldfold).ytrain_net = ycomtr.(fieldfold1).(fieldfold); % Save train predictions
    selected_models.(fieldfold1).(fieldfold).yTest = yTest.(fieldfold1).(fieldfold); % Save actual Test
    selected_models.(fieldfold1).(fieldfold).yTrain = yTrain.(fieldfold1).(fieldfold); % Save actual Train
    selected_models.(fieldfold1).(fieldfold).Testperf = Testperf_record(i, h); % Save test performance
    selected_models.(fieldfold1).(fieldfold).Trainperf = Trainperf_record(i, h); % Save train performance
    selected_models.(fieldfold1).(fieldfold).testR2 = testR2(i, h); % Save test R^2
    selected_models.(fieldfold1).(fieldfold).trainR2 = trainR2(i, h); % Save train R^2
    selected_models.(fieldfold1).(fieldfold).testRMSE = testRMSE(i, h); % Save test RMSE
    selected_models.(fieldfold1).(fieldfold).trainRMSE = trainRMSE(i, h); % Save train RMSE
    selected_models.(fieldfold1).(fieldfold).testMAE = testMAE(i, h);
    selected_models.(fieldfold1).(fieldfold).trainMAE = trainMAE(i, h);
    % selected_models.(fieldfold1).(fieldfold).testRMSE = testRMSE_raw(i, h); % Save test RMSE_raw
    % selected_models.(fieldfold1).(fieldfold).trainRMSE = trainRMSE_raw(i, h); % Save train RMSE_raw
end

filename = sprintf('SelectedModels_h2_%d.mat', h2);
% % Save the selected models to a .mat file
save(filename, 'selected_models');
% 
disp('Selected models have been saved!');


%% Predict unseen data with 10 data
% Define the specific cases to save
cases_to_save = [
    struct('h', 17, 'i', 2);
    struct('h', 18, 'i', 1);
    struct('h', 19, 'i', 1);
    struct('h', 46, 'i', 2);
    struct('h', 49, 'i', 2);
];


input_test = input'; % Transpose input data for prediction
%fields_node = {'Node1','Node5', 'Node9', 'Node15', 'Node17', 'Node21', 'Node24', 'Node25', 'Node26', 'Node27', 'Node32', 'Node33', 'Node37', 'Node41', 'Node46', 'Node47', 'Node49', 'Node50', 'Node53', 'Node54', 'Node57', 'Node59'};

 k=1;
 f=1;
for n=1:size(cases_to_save,1)
    fieldfold=sprintf('Fold%d',cases_to_save(n).i);
    fieldfold1=sprintf('Node%d',cases_to_save(f).h);

        % Load the trained network for prediction
        trained_net = selected_models.(fieldfold1).(fieldfold).net;

        % Predict using the trained network
        y_test_predict = trained_net(input_test);

        % Store the predictions 
        predictions.(fieldfold1).(fieldfold).y_test_predict = y_test_predict;

        % Convert predicted back to raw scale
        all_predicted_raw.(fieldfold1).(fieldfold) = (y_test_predict.*outputdata_std) + outputdata_mean;
        allnorm_predicted_raw.(fieldfold1).(fieldfold) = y_test_predict;

        %Compute average values 
        s_predicted(:,k) = all_predicted_raw.(fieldfold1).(fieldfold);
        snorm_predicted(:,k) = allnorm_predicted_raw.(fieldfold1).(fieldfold);
        avg_predicted = mean(s_predicted,2);
        avgnorm_predicted = mean(snorm_predicted,2);
        k=k+1;
        f=f+1;
end
% Save predictions to a file
save('Predictions.mat', 'predictions');

% Display completion message
disp('Predictions using selected models have been saved!');


% Convert actual values back to raw scale
all_actual_raw = (data_output .* outputdata_std) + outputdata_mean;

% finding unseentestR2, unseentestRMSE

unseentestR2 = 1 - sum((all_actual_raw - avg_predicted).^2) / sum((all_actual_raw - mean(all_actual_raw)).^2);
unseentestRMSE = sqrt(mean((avgnorm_predicted - data_output).^2)); %normalized RMSE
unseentestMAE = mean(abs(avgnorm_predicted - data_output)); %normalized MAE

%% finding selectedmodelR2, selectedmodelRMSE
% Extract Train R2 and Train RMSE correctly
trainR2_values = [];
trainRMSE_values = [];
trainMAE_values = [];
testR2_values = [];
testRMSE_values = [];
testMAE_values = [];
trainy_values = [];


k=1;
f=1;
for n=1:size(cases_to_save,1)
    fieldfold=sprintf('Fold%d',cases_to_save(n).i);
    fieldfold1=sprintf('Node%d',cases_to_save(f).h);

        % Store train values into an array for averaging
        sel_trainR2_values(:,k) = [trainR2_values, selected_models.(fieldfold1).(fieldfold).trainR2];
        sel_trainRMSE_values(:,k) = [trainRMSE_values, selected_models.(fieldfold1).(fieldfold).trainRMSE];
        sel_trainMAE_values(:,k) = [trainMAE_values, selected_models.(fieldfold1).(fieldfold).trainMAE];
        sel_trainy_values(:,k) = (selected_models.(fieldfold1).(fieldfold).ytrain_net(1:77))';
        sel_atrainy_values(:,k) = selected_models.(fieldfold1).(fieldfold).yTrain(1:77);


         % Store test values into an array for averaging
        sel_testR2_values(:,k) = [testR2_values, selected_models.(fieldfold1).(fieldfold).testR2];
        sel_testRMSE_values(:,k) = [testRMSE_values, selected_models.(fieldfold1).(fieldfold).testRMSE];
        sel_testMAE_values(:,k) = [testMAE_values, selected_models.(fieldfold1).(fieldfold).testMAE];
        sel_testy_values(:,k) = (selected_models.(fieldfold1).(fieldfold).ypredict(1:38))';
        sel_atesty_values(:,k) = selected_models.(fieldfold1).(fieldfold).yTest(1:38);

    k=k+1;
    f=f+1;
end

% Compute the average selected model R2 and selected model RMSE
selecttrainR2 = mean(sel_trainR2_values);
selecttrainRMSE = mean(sel_trainRMSE_values);%normalized RMSE
selecttrainMAE = mean(sel_trainMAE_values);
selecttestR2 = mean(sel_testR2_values);
selecttestRMSE = mean(sel_testRMSE_values);%normalized RMSE
selecttestMAE = mean(sel_testMAE_values);%normalized RMSE
combine_test=reshape(sel_testy_values.',1,[]);%normalized
combine_train=reshape(sel_trainy_values.',1,[]);%normalized
combine_atest=reshape(sel_atesty_values.',1,[]);%normalized
combine_atrain=reshape(sel_atrainy_values.',1,[]);%normalized

ccombine_test=(combine_test.*outputdata_std) + outputdata_mean;%convertedback  
ccombine_train=(combine_train.*outputdata_std) + outputdata_mean;%convertedback
ccombine_atest=(combine_atest.*outputdata_std) + outputdata_mean;%convertedback
ccombine_atrain=(combine_atrain.*outputdata_std) + outputdata_mean;%convertedback

% Display results
disp(['Unseen Test R2: ', num2str(unseentestR2)]);
disp(['Unseen Test RMSE: ', num2str(unseentestRMSE)]);
disp(['Unseen Test MAE: ', num2str(unseentestMAE)]);
disp(['selected network Train R2: ', num2str(selecttrainR2)]);
disp(['selected network Train RMSE: ', num2str(selecttrainRMSE)]);
disp(['selected network Train MAE: ', num2str(selecttrainMAE)]);
disp(['selected network Test R2: ', num2str(selecttestR2)]);
disp(['selected network Test RMSE: ', num2str(selecttestRMSE)]);
disp(['selected network Test MAE: ', num2str(selecttestMAE)]);




