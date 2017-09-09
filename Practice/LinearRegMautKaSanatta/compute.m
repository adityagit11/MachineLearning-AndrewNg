# The death rate is to be represented as a function of other variables.
#
#    There are 60 rows of data.  The data includes:
#
#      I,   the index;
#      A1,  the average annual precipitation;
#      A2,  the average January temperature;
#      A3,  the average July temperature;
#      A4,  the size of the population older than 65;
#      A5,  the number of members per household;
#      A6,  the number of years of schooling for persons over 22;
#      A7,  the number of households with fully equipped kitchens;
#      A8,  the population per square mile; 
#      A9,  the size of the nonwhite population;
#      A10, the number of office workers;
#      A11, the number of families with an income less than $3000;
#      A12, the hydrocarbon pollution index;
#      A13, the nitric oxide pollution index;
#      A14, the sulfur dioxide pollution index;
#      A15, the degree of atmospheric moisture.
#      B,   the death rate.

close all;
clear all;
clc;

% Load dataset
data = load('dataset.txt');
x = data(:,2:16);
y = data(:,17);

% Normalise data
for i = 1 : size(x,2)
  x(:,i) = featureNormalize(x(:,i));
end

X = [ones(length(x),1 ) x];

% plot the data - but can't, You have 15 columns of data input
# We can plot x1 vs y for 1 variable data input
# We can plot x1 vs x2 for 2 variable data input

# Batch gradient descent

iteration = 300;
alpha = 0.01;
cost = zeros(iteration,1);
xItr = zeros(iteration,1);
theta = zeros(size(X,2),1);
m = length(y);

for i = 1 : iteration
  theta -= (alpha/m) * (X' * (X * theta - y));
  cost(i) -= 1 / (2 * m) * sum(((X*theta - y).^2));
  xItr(i) = i;
end

plot(xItr, (-1)*cost, 'rx', 'MarkerSize', 10);

% After plotting I realized We should keep iteration to 500 for alpha = 0.01

predict = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] *  theta

# Normal Equation
theta = pinv(X' * X) * X' * y;

predictN = [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0] *  theta

# Plotting Cost - but can't, You have 15 columns of data input