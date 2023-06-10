function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size,  num_labels, X, y, lambda)
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

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);

% You need to return the following variables correctly
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


%==== recode y output =====
Y = zeros(m, num_labels);
for k = 1 : num_labels
    Y(:, k) = (y == k);
end

% Part 1: Feedforward the neural network and return the cost in the
%         variable J.
%

for i = 1 : m
    % prediction for ith sample for NN
    h = sigmoid([1, sigmoid([1, X(i,:)] * Theta1')] * Theta2');

    for k = 1: num_labels
        J = J + (- Y(i,k) * log(h(k)) - (1 - Y(i,k)) * log(1 - h(k)));
    end
end

J = J / m;

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

DELTA_1 = 0;
DELTA_2 = 0;
for t = 1 : m
    % activations of all layers
    a1 = [1, X(t,:)];
    z2 = a1 * Theta1';
    a2 = [1, sigmoid(z2)];

    z3 = a2 * Theta2';
    a3 = sigmoid(z3);

    del3 = zeros(num_labels, 1);

    for k = 1: num_labels
        del3(k, 1) = a3(k) - Y(t,k);
    end

    del2 = Theta2(:, 2:end)' * del3 .* sigmoidGradient(z2');

    DELTA_1 = DELTA_1 + del2 * a1;
    DELTA_2 = DELTA_2 + del3 * a2;
end

DELTA_1 = DELTA_1 / m;
DELTA_2 = DELTA_2 / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% ===== regularization of cost function =====
REG1 = 0;

for j = 1 : hidden_layer_size
    for k = 1 : input_layer_size
        REG1 = REG1 + Theta1(j,k+1)^2;
    end
end

REG2 = 0;
for j = 1 : num_labels
    for k = 1 : hidden_layer_size
        REG2 = REG2 + Theta2(j,k+1)^2;
    end
end
J = J + lambda/(2*m) * (REG1+REG2);

% == gradient regularization ==

Theta1_grad = [DELTA_1(:, 1), DELTA_1(:, 2:end) + lambda/m * Theta1(:,2:end)];
Theta2_grad = [DELTA_2(:, 1), DELTA_2(:, 2:end) + lambda/m * Theta2(:,2:end)];

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
