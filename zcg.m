function [x, fx] = zcg(funObj, x0, maxIters, alpha)
%optimize based on conjugate gradient method

%prev_grad is the gradient of previous iteration
%prev_d is the optimization direction of previous direction
prev_grad = 0;
prev_d = 0;
prev_f = 0;

%grad is the gradient of current iteration
%d is the current step direction
grad = 0;
d = 0;
x = x0;

[f, grad] = funObj(x0);
d = -1 * grad;
k = 1;
values = [];

while 1

  %perform line search over the direction d;
  %alpha = backtrack(x, d, funObj, 1e-5, 0.7);

  % if k > 1
  %   alpha = min(1, 2*(prev_f -f ) ./ (grad' * grad ));
  % end
  x = x + alpha * d(:);

  %save previous gradient and direction
  prev_grad = grad;
  prev_d = d;
  prev_f = f;

  %calculate the current gradient and function value
  [f, grad] = funObj(x);

  beta = -1 * ( grad' * grad ) ./ (prev_grad' * prev_grad );
  d = -1 * grad + beta * prev_d;

  k = k + 1; 

  optCond = norm(grad);
  if norm(grad) < 1e-3
	  break;
  end

  values = [values f];
  fprintf('%6d %15.5e %15.5e %15.5e\n', k , alpha , f, optCond);

  if k > maxIters
     warning('exceed the maximum number of iterations')
     break;
  end
end
fprintf('Total of %d iteration\n', k);
plot(values);
end
