function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%


mu = sum(X)' / m;              %sum是对X按列求和的意思
temp = X' - repmat(mu,1,m);    %repmat(A,m,n)，把矩阵A复制m*n份，行为m，列为n
                               %这里也可用temp = bsxfun(@minus, X', mu)
sigma2 = sum(temp.^2,2) / m;   %sum(A,2)对矩阵按列求和






% =============================================================


end
