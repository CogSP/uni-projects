% %se la somma phi = q1 + q2 + q3 non è specificata allora ci sono infinite
% %soluzioni per questo problema 
% 
% syms q1 q2 t p_x p_y
% 
% t = 1
% p_x = 3 + 0.5*t*(0.75 - 3)/sqrt((0.75 - 3)^2 + (1.8 - 2.5)^2)
% p_y = 2.5 + 0.5*t*(1.8 - 2.5)/sqrt((0.75 - 3)^2 + (1.8 - 2.5)^2)
% L1 = 2; 
% L2 = 2; 
% L3 = 2; 
% 
% %% TO CHECK
% phi = atan2(2.5 - 1.8, 3 - 0.75) + pi/2
% 
% p_wx = p_x - L3*cos(phi)
% p_wy = p_y - L3*sin(phi)
% 
% cos_q2 = (p_wx^2 + p_wy^2 - L1^2 - L2^2) / (2*L1*L2)
% sin_q2_plus = sqrt(1 - cos_q2^2)
% sin_q2_neg = -sqrt(1 - cos_q2^2);
% 
% q2_plus = atan2(sin_q2_plus,cos_q2)
% q2_neg = atan2(sin_q2_neg,cos_q2)
% 
% sin_q1_plus = (p_wy*(L1 + L2*cos_q2) - L2*sin_q2_plus*p_wx) / (p_wx^2 + p_wy^2)
% sin_q1_neg = (p_wy*(L1 + L2*cos_q2) - L2*sin_q2_neg*p_wx) / (p_wx^2 + p_wy^2);
% 
% cos_q1_plus = (p_wx*(L1 + L2*cos_q2) + L2*sin_q2_plus*p_wy) / (p_wx^2 + p_wy^2)
% cos_q1_neg = (p_wx*(L1 + L2*cos_q2) + L2*sin_q2_neg*p_wy) / (p_wx^2 + p_wy^2);
% 
% q1_plus = atan2(sin_q1_plus,cos_q1_plus)
% q1_neg = atan2(sin_q1_neg,cos_q1_neg);
% 
% q3_plus = phi - q1_plus - q2_plus
% q3_neg = phi - q1_neg - q2_neg;
% 
% disp("First solution:");
% q_first = [q1_plus; q2_plus; q3_plus];
% disp(q_first);
% 
% disp("Second solution:");
% q_second = [q1_neg; q2_neg; q3_neg];
% disp(q_second);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear all

syms q1 q2 q3 p_x_dot_dot p_y_dot_dot real
syms q1_dot q2_dot q3_dot q1_dot_dot q2_dot_dot q3_dot_dot real

q = [q1; q2; q3];
q_dot = [q1_dot; q2_dot; q3_dot];
q_dot_dot = [q1_dot_dot; q2_dot_dot; q3_dot_dot];

px = 2*cos(q1) + 2*cos(q1 + q2) + 2*cos(q1 + q2 + q3);
py = 2*sin(q1) + 2*sin(q1 + q2) + 2*sin(q1 + q2 + q3);
phi = q1 + q2 + q3;
J = jacobian([px, py, phi], [q1, q2, q3]);
J = simplify(J, steps = 100)


J_inv_in_t_1 = double(subs(inv(J), [q1, q2, q3], [0.8057, -1.3298, 2.3965]))

p_d_dot_x = (0.5*((0.75 - 3)))/sqrt((0.75 - 3)^2 + (1.8 - 2.5)^2)
p_d_dot_y = (0.5*((1.8 - 2.5)))/sqrt((0.75 - 3)^2 + (1.8 - 2.5)^2)


%% the last component is the derivative of phi_desired
r_d_dot = [p_d_dot_x; p_d_dot_y; 0]

q_d_dot = J_inv_in_t_1 * r_d_dot

