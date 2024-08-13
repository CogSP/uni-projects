clear all
clc

syms q1 q2 q3 l1

px = l1*cos(q1) + q3*cos(q1 + q2)
py = l1*sin(q1) + q3*sin(q1 + q2)

phi = q1 + q2

J = simplify(jacobian([px, py, phi], [q1, q2, q3]), steps=100)

det_J = simplify(det(J), steps=100)

F = [0; 1.5; -4.5]

tau = - J.' * F

tau_subs_1 = subs(tau, [q1, q2, q3], [pi/2, 0, 3])


tau_subs_s = subs(tau, [q1, q2, q3, l1], [0, pi/2, 0, 0.5])