clc
clear all

%% this holds if the robot has the first angle (the fixed one, associated to the prismatic joint) set to 0. An analogous calculation can be done for alpha = pi/2, alpha = pi, alpha = -pi/2

syms q1 q2 px py L;

px = q1 + L*cos(q2);
py = L*sin(q2);

% find q2

s2 = py/L
c2_pos = + sqrt(1 - s2)
c2_neg = - sqrt(1 - s2)

q2_pos = atan2(s2, c2_pos)
q2_neg = atan2(s2, c2_neg)


% find q1

q1_pos = px - L*c2_pos
q1_neg = py - L*c2_neg



%% display the solution

q_first = [q1_pos; q2_pos;]
q_second = [q1_neg; q2_neg;]
