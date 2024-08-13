clc
clear all

syms q1 q2 q3


J = [-sin(q1)*(cos(q2) + cos(q2 + q3)), -cos(q1)*(sin(q2) + sin(q2 + q3)), -cos(q1)*sin(q2 + q3);
     cos(q1)*(cos(q2) + cos(q2 + q3)),  -sin(q1)*(sin(q2) + sin(q2 + q3)), -sin(q1)*sin(q2 + q3);
     0, cos(q2) + cos(q2 + q3), cos(q2 + q3);]

det_J = simplify(det(J), steps=100)

% J_qs1 = subs(J, q3, pi)
% 
% J_rank_qs1 = rank(J_qs1)
% 
% null_space_qs1 = simplify(null(J_qs1), steps=100)
% range_space_qs1 = simplify(colspace(J_qs1), steps=100)

J_qs2 = subs(J, q3, 0)

J_rank_qs2 = rank(J_qs2)

null_space_qs2 = simplify(null(J_qs2), steps=100)
range_space_qs2 = simplify(colspace(J_qs2), steps=100)

% J_q_non_singular = subs(J, [q1, q2, q3], [0, 0, pi/2])
% 
% J_rank_non_singular = rank(J_q_non_singular)
% 
% null_space_non_singular = simplify(null(J_q_non_singular), steps=100)
% range_space_non_singular = simplify(colspace(J_q_non_singular), steps=100)



R_0_1_T = [cos(q1), sin(q1), 0;
         -sin(q1), cos(q1), 0;
         0, 0, 1;]

J_in_frame_1 = simplify(R_0_1_T * J, steps=100)

J_in_frame_1_qs2 = simplify(subs(J_in_frame_1, q3, 0), steps=100)

J_in_frame_1_qs2_rank = rank(J_in_frame_1_qs2)

null_space_J_in_frame_1_qs2 = simplify(null(J_in_frame_1_qs2), steps=100)
range_space_J_in_frame_1_qs2 = simplify(colspace(J_in_frame_1_qs2), steps=100)

v_s = [-1; 1; 0]
q_s = [pi/2; 0; 0]

J_qs = subs(J, [q1, q2, q3], q_s.')

rank_J_qs = rank(J_qs)

q_s_dot = pinv(J_qs)*v_s
