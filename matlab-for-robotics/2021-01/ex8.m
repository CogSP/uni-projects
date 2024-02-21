clc
clear all

syms R px_0 py_0 beta x y q1 q2 l1 l2 p_x p_y

eqns = [x^2 + y^2 == R^2, (y-py_0) == (x-px_0)*tand(beta)];

solutions = solve(eqns, [x, y])

solution_x = double(subs(solutions.x, [px_0, py_0, beta, R], [-0.8, 1.1, -20, 0.9]))

solution_y = double(subs(solutions.y, [px_0, py_0, beta, R], [-0.8, 1.1, -20, 0.9]))

%% solution 1 should be the one we are looking for -> solution 1 is p1, so the initial position of the target
solution_1 = [solution_x(1); solution_y(1)]
solution_2 = [solution_x(2); solution_y(2)]


%% we don't need to comptue q_in since it's just (pi, 0)

q_in = [pi; 0]


%% we need to find q_fin, corresponding to p_rv


c_2 = (p_x^2 + p_y^2 - (l1^2 + l2^2)) / (2 * l1 * l2)
s_2_first_solution = sqrt(1 - c_2^2)
s_2_second_solution = -sqrt(1 - c_2^2)



%% this is for finding q_fin corresponding to p_rv
q_2_first_solution_for_final = double(subs(atan2(s_2_first_solution, c_2), [l1, l2, p_x, p_y], [0.5, 0.4, 0.3708, 0.6739]))
q_2_second_solution_for_final = double(subs(atan2(s_2_second_solution, c_2), [l1, l2, p_x, p_y], [0.5, 0.4, 0.3708, 0.6739]))


q_1_first_solution_for_final = double(subs((atan2(p_y, p_x) - atan2(l2*s_2_first_solution, l1 + l2*c_2)), [l1, l2, p_x, p_y], [0.5, 0.4, 0.3708, 0.6739]))
q_1_second_solution_for_final = double(subs((atan2(p_y, p_x) - atan2(l2*s_2_second_solution, l1 + l2*c_2)), [l1, l2, p_x, p_y], [0.5, 0.4, 0.3708, 0.6739]))


q_fin_solution_1 = [q_1_first_solution_for_final; q_2_first_solution_for_final]
q_fin_solution_2 = [q_1_second_solution_for_final; q_2_second_solution_for_final]


%% why does the professor choose the negative solution?

