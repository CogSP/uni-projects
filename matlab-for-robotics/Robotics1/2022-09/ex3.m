clear all
clc

syms q1 q2 q3 L 

r = [q2*cos(q1) + L*cos(q1 + q3);
     q2*sin(q1) + L*sin(q1 + q3);
     q1+ q3]

J = simplify(jacobian(r, [q1, q2, q3]), steps=100)

rank_J = rank(J)

det_J = simplify(det(J), steps=100)

J_s = subs(J, [q2], [0])

range_space_singularity = simplify(colspace(J_s), steps=100)

null_space = simplify(null(J), steps=100)
null_space_singularity = simplify(null(J_s), steps=100)

J_last_question = simplify(subs(J, [L, q1, q2, q3], [1, pi/2, 1, 0]), steps=100)
rank(J_last_question)

null_last_question = null(J_last_question)