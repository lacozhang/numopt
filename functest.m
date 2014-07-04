function [val, grad] = functest(x)

  val = rosebork(x);
  grad = rosegrad(x);

endfunction
