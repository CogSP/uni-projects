clear all
clc;
syms L M N q1 q2 q3 real;
L = 0.5;
M = 0.5;
N = 0.5;
pd = [0.3; -0.3; 0.7];
%initial_guess = (-pi/4, pi/4, pi/4)

k = 10;

eps = 1e-3;

p = [L*cos(q1)+N*cos(q1+q2)*cos(q3);
     L*sin(q1)+N*sin(q1+q2)*cos(q3);
     M + N*sin(q3);
    ];


% Newton Method
J = jacobian(p, [q1 q2 q3]);
J_inv = inv(J);

%qk = [0.6981; -1.5708]  % initial guess
qk = [-pi/4; pi/4; pi/4];


for i = 1:k
    J_inv_num = double(subs(J_inv, [q1 q2 q3], [qk(1) qk(2) qk(3)] ));
    p_qk = double(subs(p, [q1 q2 q3], [qk(1) qk(2) qk(3)]));
    qk = qk + J_inv_num*(pd - p_qk)
    if norm(pd-p_qk)<eps
        break
    end
    if i == k
        disp("NO convergence");
        disp(p_qk)
    end
end

disp('Error')
disp(norm(pd-p_qk))

