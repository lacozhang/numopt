Q = [2 0; 0 4];
q = [2, 3]';
A = [1, 2; 2, 1; 0, 1];
a = [8, 10, 3];
lb = [0, 0];
ub = [inf; inf];
ptions=optimset('quadprog');
options=optimset('LargeScale','off');
[xsol,fsolve,exitflag,output]=quadprog(Q,q,A,a,[],[],lb,ub,[],options);