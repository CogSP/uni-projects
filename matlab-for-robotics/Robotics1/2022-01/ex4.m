clear all

%% DH transformation matrices and direct kinematics 

clear all
clc

%% Define symbolic variables

syms alpha d d1 d2 d4 d6 a a1 a2 a3 a4 theta q1 q2 q3 q4 q5 q6 K D psi

%% number of joints 

N=3;

%% Insert DH table of parameters

DHTABLE = [ -pi/2  0    K    q1;
            pi/2   q2   0    0;
            0      0    d2   q3;];

         
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

disp('Printing the single A_i matrices:')

for i = 1:N
    % subs use alpha, a, d, and theta extracted from the DHTABLE 
    alpha = DHTABLE(i,1);
    d = DHTABLE(i,2);
    a = DHTABLE(i,3);
    theta = DHTABLE(i,4);
    A{i} = subs(TDH);
    disp(i)
    disp(A{i})
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


R_0_1 = A_0_1(1:3, 1:3);
R_0_2 = A_0_2(1:3, 1:3);
R_0_3 = A_0_3(1:3, 1:3);
R_1_2 = A{2}(1:3, 1:3);
R_2_3 = A{3}(1:3, 1:3);
R_3_E = [cos(pi/2), cos(pi/2 + psi), cos(psi);
         cos(-pi/2), cos(psi), cos(-pi/2 + psi);
         cos(-pi), cos(-pi/2), cos(pi/2);]
R_0_E = R_0_1 * R_1_2 * R_2_3 * R_3_E;

f_RFE = [0; -1; -2];
m_RFE = [2; 0; 0];


f_RF0 = simplify(R_0_E * f_RFE, steps=10);
m_RF0 = simplify(R_0_E * m_RFE, steps=10);

f_RF0_subs = double(subs(f_RF0, [q1, q2, q3, psi], [pi/2, -1, 0, -atan(1)]));
m_RF0_subs = double(subs(m_RF0, [q1, q2, q3], [pi/2, -1, 0]));



%% Second question

% Computing the Geometric Jacobian
% Extract the last column and the first three rows
p_01 = A_0_1(1:3, end);
p_02 = A_0_2(1:3, end);
p_03 = A_0_3(1:3, end);


p_0_E = p_03;
p_1_E = p_03 - p_01;
p_2_E = p_03 - p_02;


R_0_1 = A_0_1(1:3, 1:3);
R_0_2 = A_0_2(1:3, 1:3);
R_0_3 = A_0_3(1:3, 1:3);


z_0 = [0;
       0;
       1];

z_1 = simplify(R_0_1*z_0, Steps=10);
z_2 = simplify(R_0_2*z_0, Steps=10);
z_3 = simplify(R_0_3*z_0, Steps=10);

p_z_0 = simplify(cross(z_0, p_0_E), steps = 10);
p_z_1 = simplify(cross(z_1, p_1_E), steps = 10);
p_z_2 = simplify(cross(z_2, p_2_E), steps = 10);

J_L_A = [p_z_0, z_1, p_z_2;
         z_0, [0;0;0], z_2];

J_L_A_subs = double(subs(J_L_A, [q1, q2, q3, K, d2], [pi/2, -1, 0, 1, sqrt(2)]));


%% now we want the Jacobian expressed in frame RF_E

R_0_E_subs = double(subs(R_0_3 * R_3_E, [q1, q2, q3, psi], [pi/2, -1, 0, -atan(1)]));

R_of_zeros = [0, 0, 0;
              0, 0, 0;
              0, 0, 0;];

J_L_A_RF_E_subs = [R_0_E_subs.', R_of_zeros;
              R_of_zeros, R_0_E_subs.';] * J_L_A_subs


balanced_tau_in_RFE = -J_L_A_RF_E_subs.' * [f_RFE; m_RFE]
balanced_tau_in_RF0 = -J_L_A_subs.' * [f_RF0_subs; m_RF0_subs]


