
syms p_x p_y a b q1 q2

c_2 = (p_x^2 + p_y^2 - (a^2 + b^2)) / (2 * a * b)
s_2_first_solution = sqrt(1 - c_2^2)
s_2_second_solution = -sqrt(1 - c_2^2)

q_2_first_solution = double(subs(atan2(s_2_first_solution, c_2), [a, b, p_x, p_y], [1, 0.6, 0, 0.6]))
q_2_second_solution = double(subs(atan2(s_2_second_solution, c_2), [a, b, p_x, p_y], [1, 0.6, 0, 0.6]))


q_1_first_solution = double(subs((atan2(p_y, p_x) - atan2(b*s_2_first_solution, a + b*c_2)), [a, b, p_x, p_y], [1, 0.6, 0, 0.6]))
q_1_second_solution = double(subs((atan2(p_y, p_x) - atan2(b*s_2_second_solution, a + b*c_2)), [a, b, p_x, p_y], [1, 0.6, 0, 0.6]))


p = [a*cos(q1) + b*cos(q1 + q2); a*sin(q1) + b*sin(q1 + q2); q1 + q2;];
J = simplify(jacobian(p, [q1, q2]), steps = 100)
