clc;
syms q1 q2 real;
l1 = 0.5;
l2= 0.4;
pd = [2+sqrt(2)/2; sqrt(2)/2];
k= 15;
a=0.5;
eps = 1e-4;
p = [l1*cos(q1)+l2*cos(q1+q2);
     l1*sin(q1)+l2*sin(q1+q2)];
%close form solution

c2 = (pd(1)^2 + pd(2)^2 - (l1^2 + l2^2))/(2*l1*l2)
s2 = sqrt(1-c2^2) %positive solution
q2_sol = atan2(s2, c2);
q1_sol = atan2(pd(2),pd(1)) - atan2(l2*s2, l1+l2*c2);
qa = [q1_sol; q2_sol]

s2 = -sqrt(1-c2^2); %negative solution
q2_sol = atan2(s2, c2); 
q1_sol = atan2(pd(2),pd(1)) - atan2(l2*s2, l1+l2*c2);
qb = [q1_sol; q2_sol]

J = jacobian(p, [q1 q2]);
J_t = transpose(J);
rank_J_t = rank(J_t)

%qk = qa + [0.2;0]; %q0a
qk = qb + [0;0.2]; %qba
for i= 1:k
    J_t_num = double(subs(J_t, [q1 q2], [qk(1) qk(2)] ));
    rank_J_t_num = rank(J_t_num)
    p_qk = double(subs(p, [q1 q2], [qk(1) qk(2)]));
    if rank_J_t_num < rank_J_t
    	error_vector = (pd - p_qk)
    	velocity = J_t_num * error_vector
    	if velocity == [0; 0;]
    		break  
    	end
    end
    qk = qk + a*J_t_num*(pd - p_qk);
    if norm(pd-p_qk)<eps
        break
    end
    if i == k
        disp("NO convergence");
    end
end
qk
