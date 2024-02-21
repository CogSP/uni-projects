clear all
clc

syms alpha d d1 a a1 a2 a3 theta q1 q2 q3

%% number of joints 
N=3;

%assume(a1, 'positive')
%assume(a2, 'positive')
%assume(a3, 'positive')
%assume(d1, 'positive')


%% PAY ATTENTION TO THE POSITION OF
%% a and d: a is the second column
%% d the third!
DHTABLE = [ pi/2,   a1,  d1,  q1;
            0,      a2,  0,   q2;
            pi/2    a3,  0    q3; ];

         
TDH = [ cos(theta) -sin(theta)*cos(alpha)  sin(theta)*sin(alpha) a*cos(theta);
        sin(theta)  cos(theta)*cos(alpha) -cos(theta)*sin(alpha) a*sin(theta);
          0             sin(alpha)             cos(alpha)            d;
          0               0                      0                   1];

A = cell(1,N);

for i = 1:N 
    alpha = DHTABLE(i,1);
    a = DHTABLE(i,2);
    d = DHTABLE(i,3);
    theta = DHTABLE(i,4);
    A{i} = subs(TDH);
    %disp(i)
    %disp(A{i})
end


T = eye(4);

for i=1:N 
    T = T*A{i};
    T = simplify(T);
end

T0N = T;

p = T(1:3,4);

n = T(1:3,1);

s = T(1:3,2);

a = T(1:3,3);

A_0_1 = A{1};

A_0_2 = A{1} * A{2};

A_0_3 = A{1} * A{2} * A{3};


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

z_1 = simplify(R_0_1*z_0, Steps=100);
z_2 = simplify(R_0_2*z_0, Steps=100);
z_3 = simplify(R_0_3*z_0, Steps=100);

p_z_0 = simplify(cross(z_0, p_0_E), steps = 100);
p_z_1 = simplify(cross(z_1, p_1_E), steps = 100);
p_z_2 = simplify(cross(z_2, p_2_E), steps = 100);


J_L_A = [p_z_0, p_z_1, p_z_2;
         z_0, z_1, z_2;]


rotation_mid_frame_1 = [R_0_1, [0, 0, 0; 
                                  0, 0, 0; 
                                  0, 0, 0;];
                        [0, 0, 0; 
                         0, 0, 0; 
                         0, 0, 0;], R_0_1]


rotation_mid_frame_2 = [R_0_2, [0, 0, 0; 
                                  0, 0, 0; 
                                  0, 0, 0;];
                        [0, 0, 0; 
                         0, 0, 0; 
                         0, 0, 0;], R_0_2]



rotation_mid_frame_3 = [R_0_3, [0, 0, 0; 
                                  0, 0, 0; 
                                  0, 0, 0;];
                        [0, 0, 0; 
                         0, 0, 0; 
                         0, 0, 0;], R_0_3]



J_L_A_mid_frame_1 = simplify(rotation_mid_frame_1.' * J_L_A, steps=100)
J_L_A_mid_frame_2 = simplify(rotation_mid_frame_2.' * J_L_A, steps=100)
J_L_A_mid_frame_3 = simplify(rotation_mid_frame_3.' * J_L_A, steps=100)


J_L_frame_0 = J_L_A(1:3, 1:3)
J_A_frame_0 = J_L_A(4:6, 1:3)

det_J_L_frame_0 = simplify(det(J_L_frame_0), steps=100)
det_J_A_frame_0 = simplify(det(J_A_frame_0), steps=100)


J_L_frame_1 = J_L_A_mid_frame_1(1:3, 1:3)
J_A_frame_1 = J_L_A_mid_frame_1(4:6, 1:3)


det_J_L_frame_1 = simplify(det(J_L_frame_1), steps=100)
det_J_A_frame_1 = simplify(det(J_A_frame_1), steps=100)


J_L_frame_2 = J_L_A_mid_frame_2(1:3, 1:3)
J_A_frame_2 = J_L_A_mid_frame_2(4:6, 1:3)


det_J_L_frame_2 = simplify(det(J_L_frame_2), steps=100)
det_J_A_frame_2 = simplify(det(J_A_frame_2), steps=100)


J_L_frame_3 = J_L_A_mid_frame_3(1:3, 1:3)
J_A_frame_3 = J_L_A_mid_frame_3(4:6, 1:3)


det_J_L_frame_3 = simplify(det(J_L_frame_3), steps=100)
det_J_A_frame_3 = simplify(det(J_A_frame_3), steps=100)



rank_J_A = rank(J_A_frame_0)

range_space_J_A = simplify(colspace(J_A_frame_0), steps=100)


%% from 3_p_D to 0_p_D

syms D

% made homogeneous
p_3_D = [0; 0; D; 1];

p_0_D = simplify(A_0_3 * p_3_D, steps=100)

% we remove the last element, WAS IT NECESSARY TO ADD
% IT IN THE FIRST PLACE?
p_0_D = simplify(p_0_D(1:3))


%% finding v_D = \dot{p_D}

J_analytical = simplify(jacobian(p_0_D, [q1, q2, q3]), steps=100) 

ciao = double(subs(J_analytical, [q1, q2, q3, a1, a2, a3, d1, D], [0, pi/2, 0, 0.04, 0.445, 0.04, 0.33, 0.52]))

syms q1_dot q2_dot q3_dot

q_dot = [q1_dot; q2_dot; q3_dot;];

p_0_D_dot = double(subs(J_analytical*q_dot, [q1, q2, q3, q1_dot, q2_dot, q3_dot, a1, a2, a3, d1, D], [0, pi/2, 0, 0, pi/4, pi/2, 0.04, 0.445, 0.04, 0.33, 0.52]))
