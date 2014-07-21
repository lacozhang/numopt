function [x, fval] = znewton(f, x0)
%

  values = [];
  
  [f0 , current_grad]  = f(x0);

  count = 0;

  if norm(current_grad) < 1e-3
     x = x0;
     fval = f(x0);
  else
      x = x0;
      x = x(:);
      while 1

	H = rosehess(x);

	[v, d] = eigs(H);
	
	min_eigen_value = min(diag(d));

	if min_eigen_value < 1e-2
	  H = H + (1e-2 - min_eigen_value)*eye(size(H));
	end

	direction =  -H \ current_grad;

	alpha = backtrack(x, direction, f, 1e-5, 0.9);

	x = x + alpha * direction(:);

	[fx, current_grad] = f(x);
	if norm(current_grad) < 1e-3
	   break;
	end

	count = count + 1;
	values = [values, norm(current_grad)];
      end
  end

[fval, fgrad] = f(x);
plot(values);
fprintf('%d iterations used\n', count);
endfunction
