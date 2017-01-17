function [fval, optCond] = zlbfgs(funObj, x0, maxCnt, maxIters, \
				  alpha)
%optimization based on L-BFGS, used to optimize funObj
% [f, g] = funObj will return function value and gradient at point x
% x0     : initial point
% maxCnt : the last k iterations used to construct Hessien
% approximation
% maxIters : maximum number of iterations
% alpha : initial step size:

if nargin < 2
   error('Must supply function and initial point x0')
end

if nargin < 3
   maxCnt = 10;
end

if nargin < 4
   maxIters = 200;
end

if nargin < 5
   alpha = 1e-4;
end

funEvals = 1;
[f, g] = funObj(x0);
w = x0;

values = [];

while 1
      
      if funEvals > 1
	 g_diff = g - g_old;
	 [S, Y] = lbfgsUpdate(S, Y, -alpha*d, g_diff, maxCnt);
	 H0 = -alpha * (d' * g_diff) ./ (g_diff' * g_diff);
	 
	 d = lbfgs(g, S, Y, H0);
	 alpha = 1;
      else
	S = zeros(length(w), 0);
	Y = zeros(length(w), 0);
	H0 = 1;
	d = -g;
      end
      
      alpha = backtrack(w, d, funObj, 1e-5, 0.9);
      w = w + alpha * d;

      %update parameter
      g_old = g;

      [f, g] = funObj(w);
      values = [values f];
      optCond = norm(g, 'inf');

      fprintf('%6d %15.5e %15.5e %15.5e\n', funEvals, alpha, f, optCond);
      if optCond < 1e-2
	 break;
      end

      if funEvals >  maxIters
	 warning('exceed the maximum number of iterations')
	 break;
      end

      funEvals = funEvals + 1;
end

[fval] = funObj(w);
plot(values);
end
