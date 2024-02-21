clc
clear all

syms L q1 q2 q3

assume(L, 'positive')

r = [q1 + L*cos(q2) + L*cos(q2 + q3);
     L*sin(q2) + L*sin(q2 + q3);
     q2 + q3;]

J = simplify(jacobian(r, [q1, q2, q3]))

det_J = simplify(det(J), steps=100)

qs = [0; pi/2; 0]

J_qs = subs(J, [q1, q2, q3], [qs(1), qs(2), qs(3)])

rank_J_qs = rank(J_qs)

null_space = simplify(null(J_qs), steps=100)

range_space = simplify(colspace(J_qs), steps=100)

% choosing r_1_dot = [1; 0; 0];
r_1_dot = [1; 0; 0];
q_1_dot = simplify(pinv(J_qs)*r_1_dot, steps=100)

% unfeasible task velocity
complementary_range_space = simplify(null(J_qs.'), steps=100)


% (0; 0; 0) = J^T * F 
null_space_F_t = simplify(null(J_qs.'), steps=100)
