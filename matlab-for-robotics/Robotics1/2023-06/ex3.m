%% DH transformation matrices and direct kinematics 

clear all
clc

%% Define symbolic variables

syms alpha d d1 d2 d4 d6 a a1 a2 a3 a4 theta q1 q2 q3 q4 q5 q6

%% number of joints 

N=4;

%% Insert DH table of parameters

DHTABLE = [ 0     d1    a1   q1;
            pi/2  d2    0    q2;
            0     0     a3   q3;
            0     0     a4   q4];

         
%% Build the general Denavit-Hartenberg trasformation matrix

TDH = [ cos(theta) -sin(theta)*cos(alpha)  sin(theta)*sin(alpha) a*cos(theta);
        sin(theta)  cos(theta)*cos(alpha) -cos(theta)*sin(alpha) a*sin(theta);
          0             sin(alpha)             cos(alpha)            d;
          0               0                      0                   1];

%% Build transformation matrices for each link
% First, we create an empty cell array

A = cell(1,N);

% For every row in 'DHTABLE' we substitute the right value inside
% the general DH matrix

%disp('Printing the single A_i matrices:')

for i = 1:N
    % subs use alpha, a, d, and theta extracted from the DHTABLE 
    alpha = DHTABLE(i,1);
    d = DHTABLE(i,2);
    a = DHTABLE(i,3);
    theta = DHTABLE(i,4);
    A{i} = subs(TDH);
    %disp(i)
    %disp(A{i})
end

% at the end of this loop A should contains, for each i in N
% in position i the tranformation matrix from frame i to i + 1


%% Direct kinematics

disp('Direct kinematics of the robot in symbolic form (simplifications may need some time)')

disp(['Number of joints N=',num2str(N)])

% Note: 'simplify' may need some time

% eye(n) returns the n-by-n identity matrix
T = eye(4);

for i=1:N 
    T = T*A{i};
    T = simplify(T);
end

% output TN matrix
% T-O-N because it goes from frame 0 to frame N
T0N = T;

% output ON position
% this is the position of the end-effector
% expressed in the base frame 0
p = T(1:3,4);

% output xN axis

n=T(1:3,1);

% output yN axis

s=T(1:3,2);

% output zN axis

a=T(1:3,3);

A_0_1 = A{1};

A_0_2 = A{1} * A{2};

A_0_3 = A{1} * A{2} * A{3};

A_0_4 = simplify(A{1} * A{2} * A{3} * A{4}, steps = 100);

% Extract the last column and the first three rows
p_01 = A_0_1(1:3, end);
p_02 = A_0_2(1:3, end);
p_03 = A_0_3(1:3, end);
p_04 = A_0_4(1:3, end);

p_0_E = p_04;
p_1_E = p_04 - p_01;
p_2_E = p_04 - p_02;
p_3_E = p_04 - p_03;

R_0_1 = A_0_1(1:3, 1:3);
R_0_2 = A_0_2(1:3, 1:3);
R_0_3 = A_0_3(1:3, 1:3);
R_0_4 = A_0_4(1:3, 1:3);

z_0 = [0;
       0;
       1];
z_1 = simplify(R_0_1*z_0, Steps=100);
z_2 = simplify(R_0_2*z_0, Steps=100);
z_3 = simplify(R_0_3*z_0, Steps=100);

p_z_0 = simplify(cross(z_0, p_0_E), steps = 100);
p_z_1 = simplify(cross(z_1, p_1_E), steps = 100);
p_z_2 = simplify(cross(z_2, p_2_E), steps = 100);
p_z_3 = simplify(cross(z_3, p_3_E), steps = 100);

J_L_A = [p_z_0, p_z_1, p_z_2, p_z_3;
         z_0, z_1, z_2, z_3]

det_J_L_A = simplify(det(J_L_A.' * J_L_A), steps=100)
% solutions=solve(det_J_L_A==0,[q1 q2 q3 q4],'Real',true, ReturnConditions=true)
% solutions.q1
% solutions.q2
% solutions.q3
% solutions.q4
% solutions.conditions

zeros_3_by_3 = [0,0,0;
                0,0,0;
                0,0,0;];

R_0_2_T = [R_0_2.', zeros_3_by_3;
           zeros_3_by_3, R_0_2.'];

J_2 = simplify(R_0_2_T * J_L_A, steps=1000)
det_J_L_A_2 = simplify(det(J_2.' * J_2),steps=100)
% solutions=solve(det_J_L_A_2==0,[q1 q2 q3 q4],'Real',true, ReturnConditions=true)
% solutions.q1
% solutions.q2
% solutions.q3
% solutions.q4
% solutions.conditions
T = [1,0,0,0;
     -1,1,0,0;
     0,0,1,0;
     0,0,-1,1;]
J_2_simplified = simplify(J_2*T,steps=1000)
det_J_L_A_2_simplified = simplify(det(J_2_simplified .' * J_2_simplified),steps=100)
solutions=solve(det_J_L_A_2==0,[q1 q2 q3 q4],'Real',true, ReturnConditions=true)
solutions.q1
solutions.q2
solutions.q3
solutions.q4

%solutions=solve(j_square==0,[q1 q2 q3 q4],'Real',true, ReturnConditions=true)
%disp(solutions.conditions)

%esercizio 2
% trovare che q0 non è singolarità
% 
% J_q0 = simplify(subs(J_L_A, [q1, q2, q3, q4], [0, 0, 0, 0]), steps = 100)
% 
% J_q0_det = det(J_q0.' * J_q0)
% %a1^2*a3^2 always != 0 so no singular
% % Ja ha la quarta riga di 0 quindi non posso generare il vettpre Vb che ha
% % la riga 4 = 1 
% %ora vediamo Va
% 
% V_a = [ 0;
%         3;
%        -3;
%         0;
%         0;
%         1];
% 
% J_qo_Va = [J_q0 , V_a]
% 
% rango_J_qo_Va = rank(J_qo_Va)
% 
% %si può generare Va perchè il vettore non è linearmente indipendente perchè
% %il rango non è cresciuto 
% 
% %terzo punto
% % q_dot = J pseudo inversa * Va
% 
% J_pseudo_inverso = pinv(J_q0)
% q0_dot = simplify(J_pseudo_inverso * V_a, steps = 100)
% 
