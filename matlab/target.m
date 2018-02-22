function [fval, fgrad] = target(A, x)

[m, n] = size(A);

if n ~= length(x)
	error('the size of the column must equal the length of x')
end

%compute function value

fval = 0;
for i=1:m
	fval = fval -  log( 1 - A(i, :) * x(:) );
end

for i=1:n
	fval = fval - log( 1 - x(i) .^ 2 );
end

fgrad = zeros(n, 1);
for i=1:m
	fgrad = fgrad + A(i,:)(:) ./ ( 1 - A(i,:) * x );
end
for i=1:n
	fgrad(i) = fgrad(i) + 2*x(i) ./ (1 - x(i) .^ 2 );
end

end
