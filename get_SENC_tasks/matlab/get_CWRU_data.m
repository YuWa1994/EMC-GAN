clc;clear;close all

% datapath, all datafiles have been converted to '.mat' format
addpath(genpath('C:\Users\yuwa\Desktop\get_SENC_tasks\CWRU-data'))

rng(0)

% ------------------Data 12k----------------------
% Each health condition contains time series acquiring from three load
% conditions, L1, L2, L3
datafile = [098, 099, 100,...    % health
            106, 107, 108,...    % Inner race, Fault diameter 0.007
            170, 171, 172,...    % Inner race, Fault diameter 0.014
            210, 211, 212,...    % Inner race, Fault diameter 0.021
            131, 132, 133,...    % Outer race, Fault diameter 0.007
            198, 199, 200,...    % Outer race, Fault diameter 0.014
            235, 236, 237,...    % Outer race, Fault diameter 0.021
            119, 120, 121,...    % Ball, Fault diameter 0.007
            186, 187, 188,...    % Ball, Fault diameter 0.014
            223, 224, 225];      % Ball, Fault diameter 0.021
        
% Each datafile extract 'num' instances, using sliding window of 1024 
% length, and 400 overlap        
num = 250;  
data = get_data(datafile, num, 400, 1024);

idx = 1:3;
X = cell(10,1);
for c = 1:10
    X{c} = cell2mat(data(idx));
    p = randperm(num*3);
    X{c} = X{c}(p,:);
    idx = idx + 3;
end
data = X;

H = data{1};     
I7 = data{2};    
I14 = data{3};
I21 = data{4};
O7 = data{5};
O14 = data{6};
O21 = data{7};
B7 = data{8};
B14 = data{9};
B21 = data{10};      

% Shuffle data
% get the shuffled data indices
train_num = 250;  % Define the number of training for each health condition
n = randperm(num*3); % generate random number
train_idx = n(1:train_num); % get the shuffled indices of training set
test_idx = n(train_num+1:end); % get the shuffled indice of test set

% Shuffle the whole dataset
for i = 1:numel(X)
    X{i} = X{i}([train_idx, test_idx],:);
end
data = cell2mat(X);

%% c1, K9
% Define the health conditions of known classes, remaining are new classes
known = [1,2,3,4]; 

% Define the labels of each health condition
datalabel = [{'Health'}, {'I07'}, {'I14'}, {'I21'},...
             {'O07'}, {'O14'}, {'O21'}, ...
             {'B07'}, {'B14'}, {'B21'}];
         
% Get the label of the SENC task
figure, 
Y = get_Streaming_data(...
    'classes', 10, 'train_num', train_num,...  % 
    'num1', train_num*3, 'num2', 500,... % num1 is the total number of each new classes, num2 is the number of samples for current new class before next new class emerges
    'known', known,...   % the labels of known classes
    'is_plot_figure', 'true',...  
    'datalabel', datalabel,... % the defined label name of each health condition, for ploting figure
    'ratio', 0.7);   % a parameter to control the proability of occurence of unknown class and already known classes.
    
% Get the indices of input data X according to SENC task labels
S = Y;  
all = train_num*3;  % 3 load conditions for each health condition
for i = 1:numel(unique(Y))
    idx = find(Y==i);
    S(idx) = (1:all) + all*(i-1);
end

% Get the indices of input data X of the SENC task
X = data(S,:);
% Get the train and test data
trainX = X(1:train_num*numel(known), :);
trainY = Y(1:train_num*numel(known));
testX = X(train_num*numel(known)+1:end, :);
testY = Y(train_num*numel(known)+1:end);

% configure plot
xlim([1000, 7500])
set(gca, 'XTickLabel', 0:1000:6500)

save(['CWRU', num2str(numel(known)), '-', num2str(10-numel(known))],...
    'trainX', 'trainY', 'testX', 'testY')

%% c1, K3
known = [1,2];
datalabel = [{'Health'}, {'B07'}, {'B14'}, {'B21'}];
figure, 
Y = get_Streaming_data(...
    'classes', 4, 'train_num', train_num,...
    'num1', train_num*3, 'num2', 500,...
    'known', known,...
    'is_plot_figure', 'true',...
    'datalabel', datalabel,...
    'ratio', 0.7);

S = Y;
for i = 1:numel(unique(Y))
    idx = find(Y==i);
    S(idx) = (1:all) + all*(i-1);
end
X = data(S,:);

trainX = X(1:train_num*numel(known), :);
trainY = Y(1:train_num*numel(known));
testX = X(train_num*numel(known)+1:end, :);
testY = Y(train_num*numel(known)+1:end);

% configure plot
xlim([500, 3000])
set(gca, 'Xtick', 500:500:size(testY,1), 'XTickLabel', 0:500:size(testY,1))

save(['CWRU', num2str(numel(known)), '-', num2str(4-numel(known))],...
    'trainX', 'trainY', 'testX', 'testY')

%% functions
function output = get_data(datafiles, sample_num, stride, dimension)
if nargin < 3, stride = sample_num; end
if nargin < 4, dimension = 1024; end

output = cell(numel(datafiles),1);
count = 0;
for i = datafiles
    count = count+1;
    
    if i < 100
        data = strcat('0',num2str(i));
    else
        data = num2str(i);
    end
    
    file = strcat('X', data, '_DE_time');
    load(data, file)
    output{count,:}= slide_window(eval(file),dimension,sample_num, stride);
    clear(file)
end
end


