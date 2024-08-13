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
