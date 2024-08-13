clc
clear all

syms l1 l2 q1 q2 q3

r = [l1*cos(q1) + q3*cos(q1 + q2);
     l1*sin(q1) + q3*sin(q1 + q2);
     q1 + q2;]

J = simplify(jacobian(r, [q1, q2, q3]), steps=100)

det_J = simplify(det(J), steps=100)

tau_balances = simplify(- J.' * [0; 1.5; -4.5;], steps=100)

tau_balances_subs_q0 = double(subs(tau_balances, [q1, q2, q3, l1], [pi/2, 0, 3, 0.5]))

tau_balances_subs_qs = double(subs(tau_balances, [q1, q2, q3, l1], [pi/2, pi/2, 3, 0.5]))

