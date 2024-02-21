clear;
clc;
syms q1 q2 t real;
assume(t>=0)

p = [q2*cos(q1);
     q2*sin(q1);];

J = jacobian(p, [q1 q2])
determinant = simplify(det(J))

T = 2;
q0 = [0; 1];
A1 = 2;
A2 = 0.5;
V1 = A1*T/4;
q1_1 = A1/2*(T/4)^2;
q1_2 = V1*(T/2) + q1_1;
q2_0 = 1;
q2_1 = -0.5*A2*(T/2)^2 +q2_0
V2 = -A2*T/2


q1_t = piecewise((t<=T/4&t>=0), 0.5*A1*t^2, (t<=3*T/4&t>=T/4),V1*(t-T/4) + q1_1, (t<=T&t>=3/4*T), -0.5*A1*(t-3/4*T)^2 + V1*(t-3/4*T)+ q1_2)
q1_vel_t = diff(q1_t,t);
q1_acc_t = diff(q1_vel_t, t);

time = 0:0.01:T;
q1_values = subs(q1_t, t, time);
q1_vel_values = subs(q1_vel_t, time);
q1_acc_values = subs(q1_acc_t, time);
figure;
plot(time, q1_values);
hold on;
plot(time, q1_vel_values);
hold on;
plot(time,q1_acc_values);
grid on;
title("q1 in time")

q2_t = piecewise((t<=T/2), -0.5*A2*t^2 + q2_0, (t>=T/2&t<=T), 0.5*A2*(t-T/2)^2 + V2*(t-T/2) + q2_1);

q2_vel_t = diff(q2_t,t);
q2_acc_t = diff(q2_vel_t, t);

time = 0:0.01:T;
q2_values = subs(q2_t, t, time);
q2_vel_values = subs(q2_vel_t, time);
q2_acc_values = subs(q2_acc_t, time);
figure;
plot(time, q2_values);
hold on;
plot(time, q2_vel_values);
hold on;
plot(time,q2_acc_values);
grid on;
title("q2 in time")

J_
%no singularity during motion!

q_mid = [double(subs(q1_t, t, T/2)); double(subs(q2_t, t, T/2))]
q = [q1_t; q2_t];
p_vel = J*q;
p_der_1 = diff(J, q1);
p_der_2 = diff(J, q2);
