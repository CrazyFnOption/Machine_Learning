function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


X = [ones(m,1), X];

item2 = sigmoid(X * Theta1');

item2 = [ones(m,1), item2];

item2 = sigmoid(item2 * Theta2');


% 现在问题又来了，为啥每一次都要去寻找每一行的最大值，因为其最大的那个值可以表现出当时最接近的那个预测值... 并且可以直接输出该预测的值

% 另外 这个函数的意思 a 代表的是输出的具体的值，p代表的是输出的具体的位置，后面1代表按照列，2代表按照行。

[a, p] = max(item2, [], 2);




% =========================================================================


end
