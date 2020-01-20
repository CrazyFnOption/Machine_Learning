function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

l = sigmoid(X * theta);

theta(1) = 0; % 注意这个地方需要找出错误 不需要去惩罚theta0 去惩罚其他的参数即可
              % 无论是计算cost function的时候 还是计算下降函数的时候都去将第一个参数初始化为0即可。

J = - (y' * log(l) + (1 - y)' * log(1 - l)) / m + lambda * (theta' * theta) / (2 * m);

              % 另外在网上看到别人的小错误是在这里用 其他变量去代替theta参数，免得其参数过于长，但是这里需要保证的是并行计算

grad = (X'* (l - y) + lambda * theta) / m;




% =============================================================

end
