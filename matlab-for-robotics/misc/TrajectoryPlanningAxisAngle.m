%axis-angle method: it doesn't work because it ignores the sign of the
%components of velocity and acceleration vectors!

clc;
R0in = [0.5 0 -sqrt(3)/2; -sqrt(3)/2 0 -0.5; 0 1 0];
R0fin = [sqrt(2)/2 -sqrt(2)/2 0; -0.5 -0.5 -sqrt(2)/2;
        0.5 0.5 -sqrt(2)/2];

Rinfin = R0in' * R0fin

sol = rotm2axang(Rinfin);

r = sol(1:3)';
thetafin = sol(4)

wfin = [3; -2; 1];
theta_velfin = norm(wfin) %modulo of thetadot!

theta_accfin = 0;
syms Ofin O_velfin T t;
A = [1 0 0 0 0 0;
     1 1 1 1 1 1;
     0 1 0 0 0 0;
     0 1 2 3 4 5;
     0 0 2 0 0 0;
     0 0 2 6 12 20;];
b = [0;Ofin; 0; O_velfin; 0; 0];
poly_sol = A\b;
a = poly_sol(1);
b = poly_sol(2);
c = poly_sol(3);
d = poly_sol(4);
e = poly_sol(5);
f = poly_sol(6);
theta_t = a+b*(t/T)+ c*(t/T)^2 + d*(t/T)^3 + e*(t/T)^4 + f*(t/T)^5;
theta_vel_t = 1/T * (b + 2*c*(t/T) + 3*d*(t/T)^2 + 4*e*(t/T)^3 + 5*f*(t/T)^4);
theta_acc_t = (1/T)^2 * (2*c + 6*d*(t/T) + 12*e*(t/T)^2 + 20*e*(t/T)^3);

theta_mid = double(subs(theta_t, [t T Ofin O_velfin], [1 1 thetafin theta_velfin]))
theta_vel_mid = double(subs(theta_vel_t, [t T Ofin O_velfin], [1 1 thetafin theta_velfin]))

w_mid = theta_vel_mid * r
R_mid = axang2rotm([r' theta_mid])
