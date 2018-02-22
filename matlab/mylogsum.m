function [ fval ] = mylogsum(b)
%compute the log sum exponential

%compute the max of each row
B = max(b, [], 2);

fval = log( sum( exp( b - repmat(B, [1 size(b, 2)] ) ), 2 ) ) + B;

end
