function [X_norm] = featureNormalize(col)

m = length(col);

mu = mean(col);
max = max(col);
min = min(col);

for i = 1 : m
  col(i) = (col(i) - mu)/(max - min);
end

X_norm = col;