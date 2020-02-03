function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



a1 = [ones(m,1), X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1), a2];
z3 = a2 * Theta2';
hx = sigmoid(z3);


% 因为此处不知一个节点，要用二进制去表示十种情况，类似于状压dp。
y_temp = zeros(m, num_labels);
for i = 1:m
    y_temp(i, y(i)) = 1;
end


% 这里有一个线性代数的小技巧，可以直接将两个矩阵直接变成两个列向量，由此就可以直接相乘得出结论，就像注射的这样
% hx = hx(:);
% y_use = y_temp(:);
% J = -(y_use' * log(hx) + (1 - y_use)' * log(1 - hx)) / m ;

% 第二种方式就是直接使用老套的循环，一次矩阵运算只能够解决一个连加符号，至于多的符号，就只能使用循环了。

for i = 1:m
    J = J + y_temp(i,:) * log(hx(i,:))' + (1 - y_temp(i,:)) * log(1 - hx(i,:))';
end 

J = J / -m;


% 正则化，惩罚参数，非常非常非常需要注意的就是 theta0 是不需要进行惩罚的，用下面这种写法是最规范的。
J = J + lambda / (2 * m) * (sum(sum(Theta1(:,2:end) .^ 2)) +sum(sum(Theta2(:,2:end) .^ 2))); 





% 反向传播算法

% 初始化 Delta
Delta2 = zeros(size(Theta2));
Delta1 = zeros(size(Theta1));

% 开始计算delta
delta3 = hx - y_temp;
delta2 = (delta3 * Theta2(:,2:end)) .* sigmoidGradient(z2);


% 再来计算Delta
Delta2 = Delta2 + delta3' * a2;
Delta1 = Delta1 + delta2' * a1;



Theta1_grad = Delta1 / m;
Theta2_grad = Delta2 / m;

Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end);
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
