syms q1 q2 q3 L q1_dot q2_dot q3_dot q1_dot_dot q2_dot_dot q3_dot_dot

q = [q1; q2; q3]
q_dot = [q1_dot; q2_dot; q3_dot];
q_dot_dot = [q1_dot_dot; q2_dot_dot; q3_dot_dot];


r = [q2*cos(q1) + L*cos(q1 + q3); q2*sin(q1) + L*sin(q1 + q3); q1 + q3;];

J = simplify(jacobian(r, [q1, q2, q3]), steps=100)

det_J = simplify(det(J), steps=100)

rank_J = rank(J)

J_qs = subs(J, q2, 0)

rank_J_qs = rank(J_qs)

range_space = simplify(colspace(J_qs), steps=100)

null_space = simplify(null(J_qs), steps=100)


%% last question
dim = size(J);
J_dot = sym(zeros(dim(1), dim(2)));
for i = 1:dim(2)
    J_dot = J_dot + diff(J, q(i)) * q_dot(i);
end
J_dot = simplify(J_dot, steps = 100)


q_dot_dot_in_r_0 = simplify(-inv(J)*(J_dot*q_dot), steps=100)

q_dot_dot_in_r_0_subs = double(subs(q_dot_dot_in_r_0, [L, q1, q2, q3, q1_dot, q2_dot, q3_dot], [1, pi/2, 1, 0, 1, -1, -1]))





