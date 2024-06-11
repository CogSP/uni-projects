clear all
clc

syms alpha d d1 d2 d4 d6 a a1 a2 a3 a4 theta q1 q2 q3 q4 q5 q6 A B C D

%% number of joints 
N=4;


%% PAY ATTENTION TO THE POSITION OF
%% a and d: a is the second column
%% d the third!

DHTABLE = [ pi/2,   B,   A,    q1;
            0,   C,  0,    q2;
            pi/2,  D,  0,   q3;
            0,      0,  q4,  0];

         
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

A_0_4 = simplify(A{1} * A{2} * A{3} * A{4}, steps = 100);

R_0_1 = A_0_1(1:3, 1:3);

p = simplify(A_0_4(1:3, 4), steps=100);

J = simplify(jacobian(p, [q1, q2 ,q3, q4]), steps=100);

J_red = simplify(subs(J, [B, D], [0, 0]), steps=100)

J_red_frame_1 = simplify(R_0_1.' * J_red)

% RATHER THAN CALCULATING THE DETERMINANT, WE CAN SEE FROM THE STRUCTURE
% OF THE MATRIX A SINGULARITY. PRECISELY WE CAN SEE THAT J_RED_FRAME_1 HAS
% A ROW OF 0 WITH JUST 1 ELEMENT != 0, WE CAN PUT THAT ELEMENT TO 0
%det_J_red = simplify(det(J_red * J_red.'), steps=100)
%det_J_red_frame_1 = simplify(det(J_red_frame_1 * J_red_frame_1.'), steps=100)

J_red_singular = simplify(subs(J_red, [q1, q2, q3, q4] , [0, pi/2, 0, 0]), steps=100)

%range_J_red_singular = simplify(colspace(J_red_singular), steps=100)
range_J_red_singular = colspace(J_red_singular)

v_star = [1; 0; 0]

q_dot_star = simplify(pinv(J_red_singular)*v_star, steps=100)