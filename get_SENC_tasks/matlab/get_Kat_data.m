clc,clear,close all

% datapath, all datafiles have been converted to '.mat' format
datapath = 'E:\Benchmark data\2 kat-BearingDataCenter\data';
addpath(genpath(datapath))

rng('default')
% files_train = [1, 9, 15, 24, 20, 19, 28, 17, 21, 31];  % K001, KI16, KI17, KI18, KA15, KA16, KA22, KB23, KB24, KB27
files_train = [1, 9, 15, 24, 20, 28]; % 1-9 , K001, KA04, KA15, KI04, KB24, KI14
datas_train = {2:11}; % 42-61
sample_num_train = {90}; 

data = get_data(files_train, datas_train, sample_num_train, datapath);

K001 = data{1};     
KI16 = data{2};    
KI17 = data{3};
KI18 = data{4};
KA15 = data{5};
KA16 = data{6};
  

train_num = 300;
all = 900;

n = randperm(all);
train_idx = n(1:train_num);
test_idx = n(train_num+1:end);

X = data;
for i = 1:numel(X)
    X{i} = X{i}([train_idx, test_idx],:);
end
data = cell2mat(X);

%% 
known = [1];
datalabel = [{'K001'}, {'KA04'}, {'KA15'}, {'KI04'},...
             {'KB24'}, {'KI14'}];
         
figure, 
Y = get_Streaming_data(...
    'classes', 6, 'train_num', train_num,...
    'num1', all, 'num2', 550,...  % 850
    'known', known,...
    'is_plot_figure', 'true',...
    'datalabel', datalabel,...
    'ratio', 0.8);  % 0.95

S = Y;
for i = 1:numel(unique(Y))
    idx = find(Y==i);
    S(idx) = (1:all) + all*(i-1);
end
% Y2 = CreatLabel(10,all*ones(10,1));
X = data(S,:);

trainX = X(1:train_num*numel(known), :);
trainY = Y(1:train_num*numel(known));
testX = X(train_num*numel(known)+1:end, :);
testY = Y(train_num*numel(known)+1:end);

% configure plot
xlim([train_num, length(testY)])
set(gca, 'XTick', train_num:800:length(testY), 'XTickLabel', 0:800:length(testY))

save(['Kat', num2str(numel(known)), '-', num2str(6-numel(known))],...
    'trainX', 'trainY', 'testX', 'testY')

%% functions
function output = get_data(wfile, wdata, sample_num, datapath)

if numel(wdata) == 1
    wdata = repmat(cell(wdata),numel(wfile),1);
end

if numel(sample_num) == 1
    sample_num = repmat(cell(sample_num),numel(wfile),1);
end

addpath(datapath)
count = 0;
output = cell(numel(wfile),1);
for i = wfile+2
    count = count+1;
    files = dir(datapath);
    filename = files(i).name;
    addpath(fullfile(datapath, filename))
    dataname = dir(fullfile(datapath, filename));
    
    X = [];
    for k = wdata{count}+2
        
        load(fullfile(datapath, filename,dataname(k).name));
        data = eval([dataname(k).name(1:end-4),'.Y(7).Data']);
        data = slide_window(data, 1024, sample_num{count}, 400);
        X = [X; data];
        
        clear(dataname(k).name(1:end-4))
    end
    
    output{count,:} = X;
end
end

