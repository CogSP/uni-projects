syms theta1 theta2 Real

eq = sin(theta1) + 2 * cos(theta1 + theta2) - 2
estremi = solve(sin(theta2) == 0.25, theta2)

first_solution = simplify(solve(subs(eq, theta2, 0) == 0 , theta1, 'Real', true) ,steps = 100)
second_solution = double(simplify(solve(subs(eq, theta2, asin(0.25)) == 0 , theta1, 'Real', true) ,steps = 100))
third_solution = double(simplify(solve(subs(eq, theta2, pi - asin(0.25)) == 0 , theta1, 'Real', true) ,steps = 100))


 
