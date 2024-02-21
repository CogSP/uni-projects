clc
clear all

syms q1 q2 p_x_dot_dot p_y_dot_dot real
syms q1_dot q2_dot q1_dot_dot q2_dot_dot real

q = [q1; q2];
q_dot = [q1_dot; q2_dot];
q_dot_dot = [q1_dot_dot; q2_dot_dot];

px = q2*cos(q1);
py = q2*sin(q1);
J = jacobian([px, py], [q1, q2]);
J = simplify(J, steps = 10)


dim = size(J);
J_dot = sym(zeros(dim(1), dim(2)));
for i = 1:dim(2)
    J_dot = J_dot + diff(J, q(i)) * q_dot(i);
end
J_dot = simplify(J_dot, steps = 10)


%% n(q, q_dot)

J_dot_q_dot = simplify(J_dot * q_dot, steps=100)



p_dot_dot = simplify(J*q_dot_dot + J_dot*q_dot, steps=100)

eqns = [p_dot_dot(1) == 0, p_dot_dot(2) == 0];

solve(eqns, [q1_dot_dot, q2_dot_dot])


det_J = simplify(det(J), steps=100)


