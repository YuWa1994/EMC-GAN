function output = slide_window(x, win, number, stride, start)
% stream: 1*n
% win: length of the window
% overlap
% output: raw is number of sample, coloum is dimension

x = x(:)';

if nargin < 5, start = 1; end
if nargin < 4 || isempty(stride), stride = win; end
if nargin < 3 || isempty(number), number = floor((length(x)-win)/stride+1); end

index = start: stride: length(x)-stride+1;
index(index > length(x)-win+1) = [];

if stride > win
    disp('The overlap should be shorter than window')
end

if isempty(number) || number > numel(index) 
    number = numel(index);
else
end

output = zeros(number,win);

for i = 1:number
    output(i,:) = x(index(i):index(i)+win-1);
end

output(all(output==0,2),:)=[];  
end


