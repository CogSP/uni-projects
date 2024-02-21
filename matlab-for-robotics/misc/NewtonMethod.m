clc;
syms q1 q2 real;
l1 = 0.5;
l2= 0.4;
pd = [0.4; -0.3];
k= 3;
eps = 1e-4;
p = [l1*cos(q1)+l2*cos(q1+q2);
     l1*sin(q1)+l2*sin(q1+q2)];
%close form solution

c2 = (pd(1)^2 + pd(2)^2 - (l1^2 + l2^2))/(2*l1*l2);
s2 = sqrt(1-c2^2); %positive solution
q2_sol = atan2(s2, c2); 
q1_sol = atan2(pd(2),pd(1)) - atan2(l2*s2, l1+l2*c2);
qa = [q1_sol; q2_sol]

s2 = -sqrt(1-c2^2); %negative solution
q2_sol = atan2(s2, c2); 
q1_sol = atan2(pd(2),pd(1)) - atan2(l2*s2, l1+l2*c2);
qb = [q1_sol; q2_sol]

J = jacobian(p, [q1 q2]);
J_inv = inv(J);
qk = qa + [0.2;0]; %q0a
qk = qb + [0;0.2]; %qba
for i= 1:k
    J_inv_num = double(subs(J_inv, [q1 q2], [qk(1) qk(2)] ));
    p_qk = double(subs(p, [q1 q2], [qk(1) qk(2)]));
    qk = qk + J_inv_num*(pd - p_qk);
    if norm(pd-p_qk)<eps
        break
    end
    if i == k
        disp("NO convergence");
    end
end
qk