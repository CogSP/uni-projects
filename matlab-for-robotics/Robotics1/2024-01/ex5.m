clc
clear all

syms x0 y0 R w t l1 l2 l3 q1 q2 q3

assume(R, 'positive')
assume(l1, 'positive')
assume(l2, 'positive')
assume(l3, 'positive')
assume(w, 'positive')

%% desired trajectory
alpha_d = w*t
px_d = x0 + R*cos(alpha_d)
py_d = y0 + R*sin(alpha_d)

r_d = [px_d; py_d; alpha_d;];


%% now we calculate the inverse kinematics of the 3R
%% using the desired trajectory as values for px, py and alpha

p_wx = px_d - l3*cos(alpha_d);
p_wy = py_d - l3*sin(alpha_d);

cos_q2 = (p_wx^2 + p_wy^2 - l1^2 - l2^2) / (2*l1*l2);
sin_q2_plus = sqrt(1 - cos_q2^2);
sin_q2_neg = -sqrt(1 - cos_q2^2);

q2_plus = atan2(sin_q2_plus,cos_q2);
q2_neg = atan2(sin_q2_neg,cos_q2);

sin_q1_plus = (p_wy*(l1 + l2*cos_q2) - l2*sin_q2_plus*p_wx) / (p_wx^2 + p_wy^2);
sin_q1_neg = (p_wy*(l1 + l2*cos_q2) - l2*sin_q2_neg*p_wx) / (p_wx^2 + p_wy^2);

cos_q1_plus = (p_wx*(l1 + l2*cos_q2) + l2*sin_q2_plus*p_wy) / (p_wx^2 + p_wy^2);
cos_q1_neg = (p_wx*(l1 + l2*cos_q2) + l2*sin_q2_neg*p_wy) / (p_wx^2 + p_wy^2);

q1_plus = atan2(sin_q1_plus,cos_q1_plus);
q1_neg = atan2(sin_q1_neg,cos_q1_neg);

q3_plus = alpha_d - q1_plus - q2_plus;
q3_neg = alpha_d - q1_neg - q2_neg;

disp("First solution:");
q_first = [q1_plus; q2_plus; q3_plus];
disp(q_first);

disp("Second solution:");
q_second = [q1_neg; q2_neg; q3_neg];
disp(q_second);


%% let's choose q_first, now we have our q_d
q_d = q_first

%% using diff to calculate q_d_dot and q_d_dot_dot

q_d_dot = diff(q_d, t)

q_d_dot_dot = diff(q_d_dot, t)


q_d_value = subs(q_d, [l1, l2, l3, x0, y0, R, w, t], [1, 1, 1, 1, 1, 0.5, 2*pi, 0.25])
q_d_dot_value = subs(q_d_dot, [l1, l2, l3, x0, y0, R, w, t], [1, 1, 1, 1, 1, 0.5, 2*pi, 0.25])
q_d_dot_dot_value = subs(q_d_dot_dot, [l1, l2, l3, x0, y0, R, w, t], [1, 1, 1, 1, 1, 0.5, 2*pi, 0.25])


%% last point
r_direct_kinematics = [l1*cos(q1) + l2*cos(q1 + q2) + l3*cos(q1 + q2 + q3);
                       l1*sin(q1) + l2*sin(q1 + q2) + l3*sin(q1 + q2 + q3);
                       q1 + q2 + q3;];

r_direct_kinematics_subs = subs(r_direct_kinematics, [q1, q2, q3, l1, l2, l3], [q_d_value(1), q_d_value(2), q_d_value(3), 1, 1, 1])


%% last point
r_direct_kinematics = [l1*cos(q1) + l2*cos(q1 + q2) + l3*cos(q1 + q2 + q3);
                       l1*sin(q1) + l2*sin(q1 + q2) + l3*sin(q1 + q2 + q3);
                       q1 + q2 + q3;];


alpha_d = w*t
px_d = x0 + R*cos(alpha_d)
py_d = y0 + R*sin(alpha_d)

r_d = [px_d; py_d; alpha_d;];

r_d_subs = subs(r_d, [q1, q2, q3, l1, l2, l3, x0, y0, w, t, R], [q_d_value(1), q_d_value(2), q_d_value(3), 1, 1, 1, 1, 1, 2*pi, 0.25, 0.5])
