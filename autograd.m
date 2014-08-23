eeeeeeeeeeefunction [f, grad_f] = autograd(x, funObj, varargin)
%compute the gradient on x based on central difference

%turning x into vector
x = x(:);
m = size(x,1);
%setup the difference
mu = 2*sqrt(1e-12)*(1+norm(x))/norm(m);

grad_f = zeros(m, 1);

for i=1:size(x,1)

    d = zeros(m, 1);
    d(i) = 1;

    fpMu = funObj(x + mu*d, varargin{:});
    fmMu = funObj(x - mu*d, varargin{:});
    grad_f(i) = (fpMu - fmMu) ./ (2*mu);
end

end
