function [f, grad_f, hess_f] = logitloss(w, X, y, lambda )
%compute the function value of logistic loss function's value, gradient and hessian for x
% X : total data sets
% w : parameter for logistic regression
% y : true labels
% lambda: regularization parameters
% logistic loss = L(y, f(x)) = log( 1 + exp(- yf(x) ) )
% derivative : d_L/d_w = - y * x ./ (1 + exp( yf(x) ) )
% hession : `

%add bias for each samples
X = [ ones(size(X, 1), 1) X ];

%each row represents a sample, so there is m samples
[m, n] = size(X);

%convert w into column vector
w = w(:);

if n ~= size(w, 1)
  error('the dimension of parameter should eaual the dimension of sample')
end

%compute f(x)
Xw = X * w;

yXw = y .* Xw;



if nargout >= 1
  f = sum(mylogsum([zeros(m, 1) -yXw])) + (lambda/2) * (w' * w);
end

if nargout >= 2
  sig = 1 ./ ( 1 + exp(yXw) );
  grad_f = lambda * w - (X') * ( y .* sig );
end

if nargout == 3
  hess_f = (X') * diag( sparse( sig .* ( 1 - sig ) ) ) * X + lambda * eye(n);
end

end
