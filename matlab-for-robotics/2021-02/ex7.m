clc;
clear all;

R0_in = [0.5, 0, -sqrt(3)/2; 
         -sqrt(3)/2, 0, -0.5; 
         0, 1, 0];

RT_fin = [sqrt(2)/2, -sqrt(2)/2, 0;
          -0.5, -0.5, -sqrt(2)/2;
          0.5, 0.5, -sqrt(2)/2];


angles=rotm2eul(R0_in, "XYZ");
alpha_in = angles(1)
beta_in = angles(2)
gamma_in = angles(3)

phi_in_value = [alpha_in; beta_in; gamma_in];


angles = rotm2eul(RT_fin, "XYZ");
alpha_fin = angles(1)
beta_fin = angles(2)
gamma_fin = angles(3)

phi_fin_value = [alpha_fin; beta_fin; gamma_fin;];

syms alpha beta gamma t real;

% T(phi) for the XYZ euler rotation
T_matrix = [[1; 0 ;0], [0 ; cos(alpha); sin(alpha)], [sin(beta); -sin(alpha) * cos(beta); cos(alpha)*cos(beta)]]
T_in = double(subs(T_matrix, [alpha, beta, gamma], [alpha_in, beta_in, gamma_in]));
T_fin = double(subs(T_matrix, [alpha, beta, gamma], [alpha_fin, beta_fin, gamma_fin]));


%% CONTROL ON SINGULARITIES
% det(T) = cos(beta) must be different from 0 so we should impose R_E,xyz
% without this singularity

%% TODO

win = [0; 0; 0];
wfin = [3; -2; 1];

% since w = T*angles_dot
phi_dot_in_value =  inv(T_in)*win
phi_dot_fin_value = inv(T_fin)*wfin

% what is the formula?
phi_dot_dot_in = [0; 0; 0];
phi_dot_dot_fin = [0; 0; 0];


%% NOW THE CONDITIONS FOR THE TRAJECTORY ARE:

% phi(0) -> our phi_in
% phi(1) -> our phi_fin
% phi_dot(0) -> our phi_dot_in
% phi_dot(1) -> our phi_dot_fin
% phi_dot_dot(0) -> our phi_dot_dot_in
% phi_dot_dot(1) -> our phi_dot_dot_fin

% so we must use a quintic

syms Ca0 Ca1 Ca2 Ca3 Ca4 Ca5 T_time phi_in phi_fin phi_dot_in T

tau = t/T_time

phi_t = phi_in + Ca0 + Ca1*tau + Ca2*tau^2 + Ca3*tau^3 + Ca4*tau^4 + Ca5*tau^5

phi_dot_t = simplify(diff(phi_t, t), Steps=100)
phi_dot_dot_t = simplify(diff(phi_dot_t, t), Steps=100)

phi_t_0 = simplify(subs(phi_t, [t], [0]), Steps = 100)
phi_t_T = simplify(subs(phi_t, [t], [T_time]), Steps = 100)

phi_dot_t_0 = simplify(subs(phi_dot_t, [t], [0]), Steps = 100)
phi_dot_t_T = simplify(subs(phi_dot_t, [t], [T_time]), Steps = 100)

phi_dot_dot_t_0 = simplify(subs(phi_dot_dot_t, [t], [0]), Steps = 100)
phi_dot_dot_t_T = simplify(subs(phi_dot_dot_t, [t], [T_time]), Steps = 100)

A = [1, 0, 0, 0, 0, 0;
     1, 1,  1 , 1 , 1 , 1;
     0, 1/T_time, 0, 0 , 0 , 0;
     0, 1/T_time , 2/T_time , 3/T_time , 4/T_time, 5/T_time;
     0, 0, 2/(T_time^2) , 0 , 0 , 0;
     0, 0, 2/(T_time^2) , 6/(T_time^2) , 12/(T_time^2), 20/(T_time^2)]

B = [0;
     phi_fin-phi_in;
     0;
     phi_dot_in;
     0;
     0]

matrix = simplify(A\B, steps = 100)
phi_t_Ca = simplify(subs(phi_t, [Ca0, Ca1, Ca2, Ca3, Ca4, Ca5], [matrix(1), matrix(2), matrix(3), matrix(4), matrix(5), matrix(6)]), steps = 100)


%% this is the symbolic result
phi_without = simplify(subs(phi_t_Ca,{phi_fin,phi_in, phi_dot_in},{phi_fin_value,phi_in_value,phi_dot_fin_value}),steps=100)
phi_trajectory_t = simplify(subs(phi_without, T_time, 1), steps=100)


% these are the tree angles alpha, beta and gamma
% of the rotation trajectory at time 0.5
phi_in_half_T = double(subs(phi_trajectory_t,t, 0.5))

% now we want to compute the rotation matrix at this time t = T/2
R_E_xyz = eul2rotm(phi_in_half_T.', 'XYZ')

%% now we want to compute the angular velocity at this time t = T/2
%% for doing so we do w(t = T/2) = T(t = T/2)*phi_dot(t = T/2)

% we need phi_dot(t = T/2)
phi_dot_in_half_T = subs(simplify(diff(phi_trajectory_t, t), steps=100), t, 0.5)

% now we compute T(t = T/2), namely T(phi(T/2))
T_in_phi_half_T = double(subs(T_matrix, [alpha, beta, gamma], [phi_in_half_T(1), phi_in_half_T(2), phi_in_half_T(3)]))

% finally, we can calculate w(t = T/2)
w_in_half_T = T_in_phi_half_T * phi_dot_in_half_T


%% NOTE: ON DE LUCA PDF THE VALUES ARE HALF, BUT PANNACCI WROTE IN THE
%% GOOGLE GROUP AND THE PROF SAID IT WAS AN ERROR