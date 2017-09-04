%{
  MACHINE LEARNING 101 - Linear regression with single variable
  Step 1: To load data
  Step 2: To plot the raw data for better visualization
  Step 3: Find the value of parameters
  Step 4: Find the minimum value of cost function
  Step 5: Visualize cost function
  Step 6: Predict the values for input new data
%}

% First we try to load data from 'ex1data1.txt'
%{
  Structure of data:
  Population of city in 10,000's               Profit in $10,000
  6.11010                                      17.59200
  5.52770                                      9.13020
%}
data = load('ex1data1.txt');

% Seperate the data marix into design/ feature matrix and output matrix
feature_matrix = data(:,1);
output_matrix = data(:,2);

% Plot the data via scatter plot
plot(feature_matrix, output_matrix, 'rx', 'MarkerSize', 10);
xlabel('Population of city in 10,000s');
ylabel('Profir in $10,000');

% Form a new matrix by concatenation where first column represent x0 = 1
X = [ones(length(feature_matrix),1) feature_matrix];

%{
  To find out the value of parameters theta0 and theta1 for which the
  value of cost function is minimum can be acieved by 2 ways:
  1. Using batch gradient technique
  2. Using Normal equation
%}
% Finding out the value of theta using Batch gradient technique
num_iteration = 1500; % choose according to your needs
alpha = 0.01;
batch_gradient_theta = zeros(2,1);
m = length(output_matrix);
cost = 0;

for i = 1 : num_iteration
  batch_gradient_theta -= (alpha/m)*(X' * (X*batch_gradient_theta - output_matrix));
  cost = 1/(2*m) * sum(((X * batch_gradient_theta - output_matrix).^2));
end

fprintf("Value of theta using batch gradient: %f\n",batch_gradient_theta);
fprintf("Value of Minimum cost using batch gradient: %f\n",cost);

% Finding out the value of theta using Normal equation
normal_equation_theta = pinv(X'*X)*X'*output_matrix;

J = 1/(2*length(output_matrix))*sum(((X*normal_equation_theta)-output_matrix).^2);

fprintf("Value of theta using normal equation: %f\n",normal_equation_theta);
fprintf("Value of Minimum Cose using normal equation: %f\n",J);

% Plotting linear regression
% Batch gradient

plot(feature_matrix, output_matrix,'rx');
hold on; %keep above plot visible
batch_gradient_linRegression = plot(feature_matrix, X*batch_gradient_theta, 'k-');

% Normal equation

%plot(feature_matrix, output_matrix,'rx');
hold on; %keep above plot visible
normal_equation_linRegression = plot(feature_matrix, X*normal_equation_theta, 'g-');
legend(batch_gradient_linRegression, 'Batch Gradient - Linear Regression', normal_equation_linRegression, 'Normal Equation - Linear Regression');


% Predict values for population sizes of 35,000 and 70,000 using batch_gradient
predict1 = [1, 3.5] *batch_gradient_theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * batch_gradient_theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);
    
% Predict values for population sizes of 35,000 and 70,000 using normal_equation
predict1 = [1, 3.5] * normal_equation_theta;
fprintf('For population = 35,000, we predict a profit of %f\n',...
    predict1*10000);
predict2 = [1, 7] * normal_equation_theta;
fprintf('For population = 70,000, we predict a profit of %f\n',...
    predict2*10000);
    
% Visualization of Cost function
theta0_vals = linspace(-10,10,100);
theta1_vals = linspace(-1,4,100);
J_vals = zeros(length(theta0_vals),length(theta1_vals));
for i = 1 : length(theta0_vals)
  for j = 1 : length(theta1_vals)
    t = [theta0_vals(i); theta1_vals(j)];
    J_vals(i,j) = 1 / (2*length(output_matrix))*sum((X*t-output_matrix).^2);
  end
end

% Because of the way meshgrids work in the surf command, we need to
% transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals';

% Surface plot
figure;
surf(theta0_vals, theta1_vals, J_vals)
xlabel('\theta_0'); ylabel('\theta_1');

% Contour plot
figure;
% Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
xlabel('\theta_0'); ylabel('\theta_1');