function [ retval ]  = linesearch(x, d, fval, grad, c1, c2)
%line search function implemented to find length alpha satisfy strong wofle conditioneeee
% x : current point 
% d : step direction
% fval : get function value of point x
% grad : get gradient of point x
% c1 : 0 < c1 < c2 < 1
% c2 :


alpha0 = 0;
alphaMax = 1e3;

%initial estimate of step length
alpha = 0.5;

%compute function value and gradient at alpha = 0 for function f(x + alpha*p)
slope = grad(x)' * d;
f0 = fval(x);

idx = 1;

prev_alpha = 0;
prev_value = 0;

while 1
      current_value = fval( x + alpha * d );

      if (current_value > ( f0 + alpha* c1 * slope )) || ( i > 1 && current_value >= prev_value)
	retval = zoom(prev_alpha, alpha, x, d, c1, c2, fval, grad);
	break	 
      end

      grad_val = grad(x + alpha * d )' * d;

      if abs(grad_val) <= -1 * c2 * slope
	 retval = alpha;
	 break;
      end

      if grad_val >= 0
	 alpha = zoom(alpha, prev_alpha, x, d, c1, c2, fval, grad);
	 break;
      end

      prev_alpha = alpha;
      prev_value = current_value;
      alpha = alpha * 5;

endwhile

endfunction

function [retval] = cubic_interpolate(alpha_a, a, a_d, alpha_b, b, b_d)
%function to find th minimizer of cubic function

	 d1 = a_d + b_d - 3 * ( a - b) ./ (alpha_a - alpha_b);
	 d2 = sign(alpha_b - alpha_a) * sqrt( d1 .^ 2 - a_d * b_d );

	 retval = alpha_b - (alpha_b - alpha_a) *( (b_d + d2 - d1) ./ (b_d - a_d + 2 * d2) );
endfunction

function [retval] = zoom(a_lo, a_hi, x, d, c1, c2, fval, grad)
%find value between [a_lo, a_hi]

	 if a_hi <= a_lo
	    error('Lower bound larger than Upper bound')
	 end

	 %compute function value and gradient at alpha = 0 for function f(x + alpha*p)
	 slope = grad(x)' * d;
	 f0 = fval(x);

	 flo = fval(x + a_lo * d);
	 fhi = fval(x + a_hi * d);

	 prev_alpha = Inf;
	 
	 idx = 1;

	 while 1

	   alpha = cubic_interpolate(a_lo, fval(x + a_lo*d), grad(x + a_lo*d)' * d, a_hi, fval(x + a_hi * d), grad(x + a_hi*d)' * d);

	   if i > 1 && abs(alpha - prev_alpha) < 1e-10
	      error('error line search')
	   end

	   current_val = fval( x + alpha * d );

	   if (current_val > f0 + c1 * alpha * slope) || current_val >= flo
	      a_hi = alpha;
	   else
	       current_grad = grad( x + alpha * d)' * d;
	       if abs(current_grad) <= -1 * c2 * slope
		 retval = alpha;
		 break;
	       end

	       if current_grad * ( a_hi - a_lo) >= 0
		  a_hi = a_lo;
	       end

	       a_lo = alpha;
	   end

	 endwhile
endfunction
