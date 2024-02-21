clc;
R0in = [0.5 0 -sqrt(3)/2; -sqrt(3)/2 0 -0.5; 0 1 0];
R0fin = [sqrt(2)/2 -sqrt(2)/2 0; -0.5 -0.5 -sqrt(2)/2;
        0.5 0.5 -sqrt(2)/2]

angles=rotm2eul(R0in, "ZYX");
alpha_in = angles(1);
beta_in = angles(2);
gamma_in = angles(3);

angles = rotm2eul(R0fin, "ZYX");
alpha_fin = angles(1);
beta_fin = angles(2);
gamma_fin = angles(3);

syms alpha beta gamma real;

T_matrix = [cos(beta)*cos(gamma) -sin(gamma) 0;
     cos(beta)*sin(gamma) cos(gamma) 0;
     -sin(beta) 0 1;];
T_in = double(subs(T_matrix, [alpha, beta, gamma], [alpha_in, beta_in, gamma_in]));
T_fin = double(subs(T_matrix, [alpha, beta, gamma], [alpha_fin, beta_fin, gamma_fin]));

wfin = [3; -2; 1];

angles_vel_fin =inv(T_fin)*wfin
syms ain afin afin_vel t T real;
A = [1 0 0 0 0 0;
     1 1 1 1 1 1;
     0 1 0 0 0 0;
     0 1 2 3 4 5;
     0 0 2 0 0 0;
     0 0 2 6 12 20];
b = [ain; afin; 0; afin_vel; 0; 0];

sol = A\b;
a = sol(1);
b = sol(2);
c = sol(3);
d = sol(4);
e = sol(5);
f = sol(6);

a_t = a + b*(t/T) + c*(t/T)^2 + d*(t/T)^3 + e*(t/T)^4 + f*(t/T)^5;
a_vel_t = 1/T * (b + 2*c*(t/T) + 3*d*(t/T)^2 + 4*e*(t/T)^3 + 5*f*(t/T)^4);
a_acc_t = (1/T)^2 * (2*c + 6*d*(t/T) + 12*e*(t/T)^2 + 20*f*(t/T)^3);

a_mid = double(subs(a_t, {t, T, ain, afin, afin_vel}, {0.5, 1, [alpha_in; beta_in; gamma_in], [alpha_fin; beta_fin; gamma_fin], angles_vel_fin}))

a_vel_mid = double(subs(a_vel_t, {t, T, ain, afin, afin_vel}, {0.5, 1, [alpha_in; beta_in; gamma_in], [alpha_fin; beta_fin; gamma_fin], angles_vel_fin}));

Tmid = double(subs(T_matrix, [alpha beta gamma], [a_mid(1) a_mid(2) a_mid(3)]));

wmid = Tmid*a_vel_mid

Rmid = eul2rotm(a_mid', "ZYX")
