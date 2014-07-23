function [x, fval] = zgradScaled(funcObj, x0, maxiters, alpha, gamma)
%gradint descent algorithm but adopt the Barzilai-Borwein step length
%compute based on its convergence.

%processing parameters
if nargin < 2
   error('you must provide the function object and start point')
end

if nargin < 3
  maxiters = 200;
  alpha = 1e-2;
  gamma = 1e-4;
end

if nargin < 4
   alpha = 1e-2;
   gamma = 1e-4;
end

[f, g] = funcObj(x0); 

count = 1;
values = [];

w = x0;
w = w(:);

while 1
  %start of the line search method

  if count > 1
     g_diff = g - g_old;
     alpha = -alpha * (g_old' * g_diff) ./ (g_diff' * g_diff);
  end

  %save old value
  f_old = f;
  g_old = g;

  %update parameter w
  w = w - alpha * g;

  [f, g] = funcObj(w);
  
  optCond = norm(g, 'inf');
	
  values = [values, f];

  fprintf('%6d %15.5e %15.5e %15.5e\n', count , alpha , f, optCond);

  count = count + 1;

  if count > maxiters
    warning('exceed the maximum number iterations')
    break;
  end
  
  if optCond < 1e-2
     break;
  end
    
end

fval = funcObj(w);
plot(values);
fprintf('%d iterations used\n', count);
end
