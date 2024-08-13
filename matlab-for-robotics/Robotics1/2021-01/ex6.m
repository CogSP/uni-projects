clc
clear all

syms a b q1 q2 q3 q4

r = [a*cos(q1) + q3*cos(q1 + q2) + b*cos(q1 + q2 + q4);
     a*sin(q1) + q3*sin(q1 + q2) + b*sin(q1 + q2 + q4);
     q1 + q2 + q4;];

J = simplify(jacobian(r, [q1, q2, q3, q4]), steps = 100)

det_J = simplify(det(J.' * J), steps = 100);

det_J = simplify(det(J * J.'), steps = 100);

J_minor_1 = J(1:3, 1:3);
det_J_minor_1 = simplify(det(J_minor_1), steps=100);
J_minor_2 = J(1:3, 2:4);
det_J_minor_2 = simplify(det(J_minor_2), steps=100);
J_minor_3 = J(1:3, [1, 3, 4]);
det_J_minor_3 = simplify(det(J_minor_3), steps=100);
J_minor_4 = J(1:3, [1, 2, 4]);
det_J_minor_4 = simplify(det(J_minor_4), steps=100);

J_q_s = subs(J, [q1, q2, q3, q4], [0, pi/2, 0, 0])

null_space = null(J_q_s)

range_space = simplify(colspace(J_q_s), steps=100)

range_space_subs = subs(range_space, [q1, q4], [0, 0])


