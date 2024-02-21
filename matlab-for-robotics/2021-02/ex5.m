clc
clear all

syms epsilon q1 q2 a

p_x = q2 * cos(q1);
p_y = q2 * sin(q1);
p = [p_x ; p_y];

epsilon = 0

q0 = [pi/4 ; epsilon];

J = jacobian([p_x, p_y], [q1 q2])

J_subs = subs(J, [q1 , q2], [pi/4 , epsilon]) 

%% det(J) = - epsilon
% So a singularity is approached when epsilon -> 0
det_J = det(J_subs)

p_d1 = [-1; 1];
p_d2 = [1; 1];


%Newton method for solution 1
J_inv = inv(J_subs);
qk = q0;
k = 1;
for i= 1:k
    J_inv_num = subs(J_inv, [q1 q2], [qk(1) qk(2)] );
    p_qk = subs(p, [q1 q2], [qk(1) qk(2)]);
    qk = qk + J_inv_num*(p_d1 - p_qk);
    if norm(p_d1 - p_qk) == 0
        break
    end
end
solution_N_1 = simplify(qk, steps = 100)

%Newton method for solution 2
J_inv = inv(J_subs);
qk = q0;
k = 1;
for i= 1:k
    J_inv_num = subs(J_inv, [q1 q2], [qk(1) qk(2)]);
    p_qk = subs(p, [q1 q2], [qk(1) qk(2)]);
    qk = qk + J_inv_num*(p_d2 - p_qk);
    if norm(p_d1 - p_qk) == 0
        break
    end
end
solution_N_2 = simplify(qk, steps = 100)


%Gradient Method for solution 1
J_t= transpose(J);
rank_J_t = rank(J_t)
qk = q0;
k = 5;
for i= 1:k
    J_t_num = subs(J_t, [q1 q2], [qk(1) qk(2)] );
    rank_J_t_num = rank(J_t_num)
    p_qk = subs(p, [q1 q2], [qk(1) qk(2)]);
    if rank_J_t_num < rank_J_t
            error_vector = (p_d1 - p_qk)
            velocity = J_t_num * error_vector
            if velocity == [0; 0;]
                break
            end
    end
    qk = qk + a*J_t_num*(p_d1 - p_qk)
end

solution_G_1 = qk, steps  = 100


%Gradient Method for solution 2
J_t = transpose(J);
qk = q0;
for i= 1:k
    J_t_num = subs(J_t, [q1 q2], [qk(1) qk(2)] );
    p_qk = subs(p, [q1 q2], [qk(1) qk(2)]);
    qk = qk + a*J_t_num*(p_d2 - p_qk);
end
solution_G_2 = simplify(qk, steps  = 100)

%% CONCLUSIONS:
% when epsilon -> 0 

% solution_N_1 goes to (inf, 0) so Newton diverges
% solution_G_1 goes to (pi/4, 0) so it simply stops

% solution_N_2 is a solution for the inverse kinematics (just one iteration
% needed). Moreover, it does not depends on epsilon
% solution_G_2 goes to (pi/2, alpha*sqrt(2)). If alpha were = 1 the
% Gradient will have find the solution too, but alpha is tipically smaller,
% so Gradient will tipically approach q* at a slower rate than Newton

%per N method la epsilon = 0 fa si che q1
%sol diverga, per G method invece dovrei calcolare il rank e il null space
%in modo da risolvere il problema e bla bla bla