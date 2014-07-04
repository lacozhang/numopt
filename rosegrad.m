function [retval] = rosegrad( x )

retval = zeros(2,1);
retval(1) = -400*(x(2) -x(1) ^ 2)*x(1) -2*(1-x(1));
retval(2) = 200*(x(2) - x(1) ^ 2);

endfunction
