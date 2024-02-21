clc
clear all

syms q1 q2

tau = [10; -5];

J = [-q2*sin(q1), cos(q1);
     q2*cos(q1), sin(q1);];

F_subs = double(subs(-(inv(J.')*tau), [q1, q2], [pi/3, 1.5]))

