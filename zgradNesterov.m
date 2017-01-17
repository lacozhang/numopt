function [x, fval] = zgrad(funcObj, x0, maxiters)
%gradint descent algorithm

if nargin < 2
   error('you must provide the function object and start point')
end

if nargin < 3
  maxiters = Inf;
  learningrate = -Inf;
end

if nargin < 4
   learningrate = -Inf;
end

[unk, current_grad] = funcObj(x0); 

count = 1;
values = [];

x = x0;

x = x(:);
x_prev = x;
x_prev_old = [];

t = 1;

while 1

  if count > 1
    tp = (1 + sqrt(1 + 4 * t^2))/2;
    x = x_prev + ((t-1)/tp)(x_prev - x_prev_old);
    t = tp;
    [func_val, func_grad] = funcObj(x);
  end

  x_prev_old = x_prev;
      
  alpha = backtrack(x, -current_grad, funcObj, 1e-5, 0.7);

  %current point
  x = x - alpha * current_grad(:);
    
  [current_value, current_grad] = funcObj(x);

  if norm(current_grad) < 1e-2
    break;
  end

  count = count + 1;
  values = [values, current_value];

  if count > maxiters
    warning('exceed the maximum number iterations')
    break;
  end
  
  fprintf('%6d %15.5e %15.5e %15.5e\n', count , alpha , current_value, norm(current_grad));
    
end

fval = funcObj(x);
plot(values);
fprintf('%d iterations used\n', count);
end
