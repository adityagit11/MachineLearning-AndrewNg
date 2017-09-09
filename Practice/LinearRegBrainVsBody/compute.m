% Dataset for Brain Weight to Body Weight
close all;
clear all;
clc;

% Load the data
data = load('Dataset.txt');
x = data(:,2:3);
y = data(:,4);

%{ 
  Data being multi feature hence needs to be normalised 
  We have three ways to normalise the data:
  1. Feature scaling
  2. Mean normalization
  3. Unit length scaling 
%}

col_1 = x(:,1);
mu_1 = mean(col_1);
max_1 = max(col_1);
min_1 = min(col_1);

for i = 1 : length(col_1)
  col_1(i) = (col_1(i) - mu_1)/(max_1 - min_1);
end

col_2 = x(:,2);
mu_2 = mean(col_2);
max_2 = max(col_2);
min_2 = min(col_2);

for i = 1 : length(col_2)
  col_2(i) = (col_2(i) - mu_2)/(max_2 - min_2);
end

X = [ones(length(x),1), col_1, col_2];

% Plot the data
plot(col_1,col_2,'rx','MarkerSize',10);

% Calculating Theta
% Method 1: Batch Gradient Descent
iteration = 1000;
alpha = 0.01;
theta = zeros(3,1);
cost = 0;
m = length(y);

for i = 1 : iteration
  theta -= (alpha/m)*(X'*(X * theta - y));
  cost = 1/(2*m) *sum(((X*theta - y).^2));
end

% Method 2: Normal Equation
thetaN = pinv(X'*X)*X'*y;

% Make predictions on theta and thetaN
predict = [1 50 50]*theta
predictN = [1 50 50]*thetaN