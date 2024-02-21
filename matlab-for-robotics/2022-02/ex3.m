
%% initialization

syms q1 q2 q3 p_x_dot_dot p_y_dot_dot real
syms q1_dot q2_dot q3_dot real

q = [q1 q2 q3];
q_dot = [q1_dot q2_dot q3_dot]

px = cos(q1) + cos(q1+q2) + cos(q1+q2+q3);
py = sin(q1) + sin(q1+q2) + sin(q1+q2+q3);



%% computing jacobian

J = jacobian([px, py], [q1, q2, q3]);
J = simplify(J, steps = 10)



%% derivative of jacobian

dim = size(J);
J_dot = sym(zeros(dim(1), dim(2)));
for i = 1:dim(2)
    J_dot = J_dot + diff(J, q(i)) * q_dot(i);
end
J_dot = simplify(J_dot, steps = 10);

J_dot = collect(J_dot, [cos(q1), cos(q1+q2), cos(q1+q2+q3), sin(q1), sin(q1+q2), sin(q1+q2+q3)])

%% n(q, q_dot)

J_dot_q_dot = J_dot * q_dot.';
J_dot_q_dot = simplify(J_dot_q_dot, steps = 10);

J_dot_q_dot = collect(J_dot_q_dot, [cos(q1), cos(q1+q2), cos(q1+q2+q3), sin(q1), sin(q1+q2), sin(q1+q2+q3)]);
J_dot_q_dot = simplify(J_dot_q_dot, steps = 10);


%J_dot_q_dot_subs = double(subs(J_dot_q_dot, [q1, q2, q3, q1_dot, q2_dot, q3_dot], [pi/4, pi/3, -pi/2, -0.8, 1, 0.2]))

%% transform J and J_dot to have 3 rows
J_with_ones = vertcat(J, ones(1, size(J, 2)))

p_dot_dot = [p_x_dot_dot; p_y_dot_dot]


p_dot_dot_minus_n = p_dot_dot - J_dot_q_dot

p_dot_dot_minus_n_with_zero = [p_dot_dot_minus_n; 0]

q_dot_dot = inv(J_with_ones) * p_dot_dot_minus_n_with_zero 

q_dot_dot_sub = double(subs(q_dot_dot, [q1, q2, q3, q1_dot, q2_dot, q3_dot, p_x_dot_dot, p_y_dot_dot], [pi/4, pi/3, -pi/2, -0.8, 1, 0.2, 1, 1]))
