syms q1 q2 q3 q4 real

%primo punto
r = [q2*cos(q1) + q4 * cos(q1+q3);
     q2 * sin(q1) + q4 * sin(q1 + q3);
     q1 + q3];

J = jacobian(r, [q1, q2, q3, q4]);
J = simplify(J, steps = 100);

J_det = simplify(det(J * J.'), steps = 100);
J_det_solve = solve(J_det == 0, [q2, q3], 'ReturnConditions', true);

%secondo punto
b = [0;
     0;
     0;
     0];

% Calcolo dello spazio nullo della matrice J
null_space = simplify(null(J), steps = 100);

null_space = null_space * q2;

J_q30_q20 = simplify(subs(J, [q2, q3], [0 , 0]), steps = 100);

null_space_J_q30_q20 = simplify(null(J_q30_q20), steps = 100);

%terzo punto
null_space_J_q30_q20_trasp = simplify(null(J_q30_q20.'), steps = 100);

%quarto punto
F_tau_0 = null(J.');

% fifth point
F_tau_0_singular = simplify(null(J_q30_q20.'), steps=100)

