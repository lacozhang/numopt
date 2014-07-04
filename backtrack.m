function [retval] = backtrack(x0, d0, f, c1, c2)
%line search algorithm based on backtracking to find point satisfy strong wolfe condition
% x0 : current point
% d0 : search direction
% f  : function will return value and gradient, [f, g] = f(x);
% 0 < c1 < c2 < 1

[f0, grad] = f(x0);
slope = grad' * d0;

if slope >= 0
   error('must be a descent direction')
end

alpha0 = 0;
alphaMax = 1e2;

alpha = 1;
dec = 0.5;
inc = 2.1;

while 1
      
      [current_val, current_grad] = f( x0 + alpha * d0);
      factor = 1;

      if current_val > ( f0 + alpha * c1 * slope)
	 factor = dec;
      else
	  current_slope = current_grad' * d0;

	  if current_slope < c2 * slope
	    factor = inc;
	  else
	      if current_slope > -c2*slope
		 factor = dec;
	      else
		  break;
	      end
	  end
      end

      if alpha < 1e-15
	 warning('too small step size')
	 %alpha = 1e-7;
	 %break;
      end

      if alpha > alphaMax
	 %alpha = 1e-7;
	 warning('too large step size')
	 %break;
      end

      alpha = alpha * factor;
endwhile
retval = alpha;
endfunction
