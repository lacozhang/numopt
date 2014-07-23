function [S,Y] = lbfgsUpdate(S,Y,s,y,memory)
if size(Y,2) < memory
    S(:,end+1) = s;
    Y(:,end+1) = y;
else
    S = [S(:,2:memory) s];
    Y = [Y(:,2:memory) y];
end