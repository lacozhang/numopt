function [retval] = rosebork( x )
%compute the function value of rosebork function

	 retval = 100 * (x(2) - x(1) ^ 2) ^ 2 + (1 - x(1)) ^ 2;
end
