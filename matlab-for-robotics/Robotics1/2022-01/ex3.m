clc
clear all

syms q1 q2 q3 p_x_dot_dot p_y_dot_dot K D real
syms q1_dot q2_dot q3_dot real

q = [q1 q2 q3];
q_dot = [q1_dot q2_dot q3_dot]

px = K*cos(q1) - q2*sin(q1) + D*cos(q1+q3);
py = K*sin(q1) + q2*cos(q1) + D*sin(q1+q3);
phi = q1 + q3

%% computing jacobian

J = jacobian([px, py, phi], [q1, q2, q3]);
J = simplify(J, steps = 100)

determinant_J = simplify(det(J), steps=100)

J_q_s = simplify(subs(J,[q1, q2, q3], [pi, 0, pi]), steps=100)

null_space_J_q_s = simplify(null(J_q_s), steps=100)

%% compute the range space

%% ????? PER ORA LO FACCIAMO A MANO



%% ex iv, solve the linear system J_q_s * q_dot = r_dot

%r_dot = [-sin(q1); cos(q1); 0];
% putting q1 = pi
r_dot = [0; 1; 0];

J_q_s_subs = simplify(subs(J_q_s, [D, K], [sqrt(2), 1]), steps=100)

q_dot = mldivide(J_q_s_subs, r_dot)


