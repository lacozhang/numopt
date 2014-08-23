function [x, fx] = zcg(f, x0)
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

[f0, grad0] = f(x0);
d = -1 * grad0;
grad = grad0;
k = 0;
values = [];
prev_f = f0;

while 1
	%perform line search over the direction d;
	alpha = backtrack(x, d, f, 1e-5, 0.7);
	x = x + alpha * d(:);

	%save previous gradient and direction
	prev_grad = grad;
	prev_d = d;

	%calculate the current gradient and function value
	[val, grad] = f(x);
	values = [values, val];

	if norm(grad) < 1e-3
		break;
	end

	beta = -1 * ( grad' * grad ) ./ ( prev_grad' * prev_grad );
	d = -1 * grad + beta * prev_d;

	prev_f = val;
	k = k + 1; 
end
fprintf('Total of %d iteration\n', k);
plot(values);
end
