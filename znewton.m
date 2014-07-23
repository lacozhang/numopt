function [x, fval] = znewton(funObj, x0, maxIters)
%Newton method for optimization
  values = [];
  
  [f0 , current_grad]  = funObj(x0);

  count = 1;

  if norm(current_grad) < 1e-3
     x = x0;
     fval = funObj(x0);
  else
      x = x0;
      x = x(:);
      while 1
	    
	[tmp_f, tmp_g, H] = funObj(x);

	[v, d] = eigs(H);
	
	min_eigen_value = min(diag(d));

	if min_eigen_value < 1e-2
	  H = H + (1e-2 - min_eigen_value)*eye(size(H));
	end
	
	%just to illustrate the use of diagnal element to find the optimal point
	%H = diag(diag(H));

	direction =  -H \ current_grad;

	alpha = backtrack(x, direction, funObj, 1e-5, 0.9);

	x = x + alpha * direction(:);

	[fx, current_grad] = funObj(x);

	norm_grad = norm(current_grad);
	fprintf('%6d %15.5e %15.5e %15.5e\n', count, alpha, fx, norm_grad);

	if norm_grad < 1e-3
	  break;
	end
 
	count = count + 1;
	values = [values, norm(current_grad)];
	
	if count > maxIters
	   fprintf('exceed the maximum number of iterations')
	end

      end
  end

  [fval, fgrad] = funObj(x);
  plot(values);
  fprintf('%d iterations used\n', count);
end
