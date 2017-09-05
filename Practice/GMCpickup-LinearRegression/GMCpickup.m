% upload and plot data
data = load('dataset.txt');
list_price = data(:,1);
best_price = data(:,2);
plot(list_price,best_price,'rx','MarkerSize',10);

% computing cost for 0 parameter
m = length(best_price);
X = [ones(m,1) list_price];
theta = zeros(2,1);
J = 1/(2*m)*sum(((X*theta)-best_price).^2);
fprintf('Value of cost for zero parameter: %f\n',J);

% Finding gradient descent using Normal Equation
theta_normal = pinv(X'*X)*X'*best_price;
J_normal = 1/(2*m)*sum(((X*theta_normal)-best_price).^2);
fprintf('\nValue of cost at global optimum using normal equation: %f\n',J_normal);
fprintf('Value of theta at global optimum using normal equation: %f\n',theta_normal);

% Finding gradient descent using Batch Gradient Algorithm
iteration = 1000;
alpha = 0.001;
J_batch = 0;
theta_batch = zeros(2,1);
for i = 1: iteration
  theta_batch -= (alpha/m)*(X'*(X*theta_batch - best_price));
  J_batch = 1/(2*m)*sum(((X*theta_batch-best_price).^2));
end
fprintf('\nValue of cost at global optimum using batch optimum: %f\n',J_batch);
fprintf('Value of theta at global optimum using batch optimum: %f\n',theta_batch);

% Plotting linear regression
plot(list_price, best_price, 'rx');
hold on;
plot(list_price, X * theta_normal);
plot(list_price, X * theta_batch);

% Predict now.
predict_1 = [1 15]*theta_normal;
fprintf('\nValue of prediction for 15 thousand dollars: %f\n',predict_1);
    
% Visualization of Cost function
theta0_vals = linspace(-10,10,100);
theta1_vals = linspace(-1,4,100);
J_vals = zeros(length(theta0_vals),length(theta1_vals));
for i = 1 : length(theta0_vals)
  for j = 1 : length(theta1_vals)
    t = [theta0_vals(i); theta1_vals(j)];
    J_vals(i,j) = 1 / (2*length(best_price))*sum((X*t-best_price).^2);
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
