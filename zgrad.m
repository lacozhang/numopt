function [x, fval] = zgrad(f, grad, x0)
%

  current_grad = grad(x0);

  count = 0;
  values = [];

  if norm(current_grad) < 1e-3
     x = x0;
     fval = f(x0);
  else
      x = x0;
      x = x(:);
      while 1

	alpha = backtrack(x, -current_grad, f, 1e-5, 0.7);

	%current point
	x = x - alpha * current_grad(:);

	[current_value, current_grad] = f(x);

	if norm(current_grad) < 1e-2
	   break;
	end

	count = count + 1;

	#values = [ values, norm(current_grad) ]; 
	values = [values, current_value];
      end
  end

fval = f(x);
plot(values);
fprintf('%d iterations used\n', count);
endfunction
