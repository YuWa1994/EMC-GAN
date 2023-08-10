
function S = get_Streaming_data(varargin)
% rng(0)
p = inputParser;
addParameter(p, 'classes', 10);  % the total number of all classes
addParameter(p, 'train_num', 250); % the number of training instances for each health condition of known classes
addParameter(p, 'num1', 750);  % % num1 is the total number of each new classes
addParameter(p, 'num2', 600);  % num2 is the number of samples for current new class before next new class emerges
addParameter(p, 'known', [1,2,3]); % the labels of known classes
addParameter(p, 'is_plot_figure', 'false');
addParameter(p, 'datalabel', 'None'); % the defined label name of each health condition, for ploting figure
addParameter(p, 'ratio', 0.5);   % a parameter to control the proability of occurence of unknown class and already known classes.

parse(p, varargin{:})
classes = p.Results.classes;
train_num = p.Results.train_num;
num1 = p.Results.num1;
num2 = p.Results.num2;
known = p.Results.known;
is_plot_figure = p.Results.is_plot_figure;
datalabel = p.Results.datalabel;
ratio = p.Results.ratio;

% the number of the whole dataset
all_num = classes*num1;

% define the total label of the training sample of known classes
S = [];
for i = known
    S = [S; ones(train_num, 1)*i];
end

current = 1 + numel(known); % define the label number of current or the first new class
x1 = known; % Labels of the initial known classes, it changes during iteration, 
            % because a new class will be regarded as a new 'known classes' 
            % after 'num2' number of this class have emerged.
while numel(S)<all_num
    if current <= classes  
        if numel(find(S==current))==num2  % After 'num2' number of the current new class have emerged.
                                          % then, the class will be taken
                                          % as a new 'known class'.
            current = current + 1;  % the next new class will emerge
            x1 = 1:current-1;       % redefine the 'current known classes'
        end
        
        % check if the number of the 'current known classes' exceeds the total number during iteration
        x1 = check_is_available(S, x1, num1);  
        
        if current <= classes
            % get the label of the next instance, there is the probability 
            % of 'ratio' that the next instance belong to new/current, and
            % probability of'1-ratio'  that the next instance belong to 
            % known classes
            new = my_rand(x1, current, ratio);  
            S = [S; new];   % store the label of the current instance
        end
    else
        x1 = 1:classes;
        x1 = check_is_available(S, x1, num1);

        new = my_rand(x1, current, 0);
        S = [S; new];  % store the label of the current instance
    end

end

%% 
if strcmp(is_plot_figure, 'true')
%     figure,
    set(gcf, 'Color', 'w')
    % define the color
    colors = [1 0 0; 
              0 1 0; 
              0 0 1; 
              0 1 1; 
              1 0 1; 
              0.8500 0.3250 0.0980; 
              0.9290 0.6940 0.1250;
              0.4940 0.1840 0.5560; 
              0.4660 0.6740 0.1880; 
              0.3010 0.7450 0.9330];
    colors = [zeros(numel(known), 3); colors];
         
    class = classes:-1:1;
    for i = 1:numel(unique(S))
        idx = find(S==i);
        stem(idx, repmat(class(i), 1, length(idx)), ...
        'LineStyle', 'None', ...
        'Marker', '.',...
        'MarkerSize', 6,...
        'Color', colors(i,:))
        hold on
    end
    legend(datalabel, 'Location', 'southwest');
    legend off
    xlabel('Number of streaming instances / time(t)')
    ylabel('Health conditions')
    set(gca, 'YTick', 1:classes, 'YTickLabel', datalabel(classes:-1:1))
    set(gca, 'FontName', 'Time New Roman')
    
end

end

%% functions
function output = my_rand(x1, x2, p)
% The probability that x2 is selected
    if numel(x1)==0
        output = x2;
    else
        x = rand(1);
        if x<=p
            output = x2;
        else
            output = randi(numel(x1));
            output = x1(output);
        end
    end
end

function output = check_is_available(S, x1, num)
output = [];
for i = x1
    if numel(find(S==i))<num
        output = [output, i];
    end
end
end



