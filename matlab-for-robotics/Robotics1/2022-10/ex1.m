clear all
clc

syms alpha d d1 d2 d4 d6 a a1 a2 a3 a4 theta q1 q2 q3 q4 q5 q6

%% number of joints 
N=3;

assume(d1, 'positive')
assume(a3, 'positive')


DHTABLE = [ pi/2   0   d1   q1;
            pi/2   0   q2   pi/2;
            0      a3   0   q3;];

         
TDH = [ cos(theta) -sin(theta)*cos(alpha)  sin(theta)*sin(alpha) a*cos(theta);
        sin(theta)  cos(theta)*cos(alpha) -cos(theta)*sin(alpha) a*sin(theta);
          0             sin(alpha)             cos(alpha)            d;
          0               0                      0                   1];

A = cell(1,N);

for i = 1:N 
    alpha = DHTABLE(i,1);
    a = DHTABLE(i,2);
    d = DHTABLE(i,3);
    theta = DHTABLE(i,4);
    A{i} = subs(TDH);
    disp(i)
    disp(A{i})
end


T = eye(4);

for i=1:N 
    T = T*A{i};
    T = simplify(T);
end

T0N = T

p = T(1:3,4)

n = T(1:3,1)

s = T(1:3,2)

a = T(1:3,3)

A_0_1 = A{1}

A_0_2 = A{1} * A{2}

A_0_3 = A{1} * A{2} * A{3}

p_03 = A_0_3(1:3, end)

J = simplify(jacobian(p_03, [q1, q2, q3]), steps=100)

det_J = simplify(det(J), steps=100)

J_in_qs1 = subs(J, q3, 0)
rank_J_qs1 = rank(J_in_qs1)
null_space_qs1 = null(J_in_qs1)
range_space_compl_qs1 = null(J_in_qs1.')

J_in_qs2 = subs(J, q3, pi)
rank_J_qs2 = rank(J_in_qs2)
null_space_qs2 = null(J_in_qs2)
range_space_compl_qs2 = null(J_in_qs2.')

J_in_qs3 = subs(J, q2, -a3*sin(q3))
rank_J_qs3 = rank(J_in_qs3)
null_space_qs3 = null(J_in_qs3)
range_space_compl_qs3 = null(J_in_qs3.')

J_in_qs4 = subs(J_in_qs1, q2, 0)
rank_J_qs4 = rank(J_in_qs4)
null_space_qs4 = null(J_in_qs4)
range_space_compl_qs4 = null(J_in_qs4.')

J_in_qs5 = subs(J_in_qs2, q2, 0)
rank_J_qs5 = rank(J_in_qs5)
null_space_qs5 = null(J_in_qs5)
range_space_compl_qs5 = null(J_in_qs5.')


