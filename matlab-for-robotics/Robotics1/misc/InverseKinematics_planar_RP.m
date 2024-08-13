clc
clear all

syms q1 q2 px py L;

px = q2*cos(q1);
py = q2*sin(q1);

% find q2

q2_pos = + sqrt(px^2 + py^2)
q2_neg = - sqrt(px^2 + py^2)

% find q1

s1 = py/q2
c1 = px/q2

q1 = atan2(s1, c1)


%% display the solution

q_first = [q1; q2_pos;];
q_second = [q1; q2_neg;];
