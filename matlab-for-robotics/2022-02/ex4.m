clc
clear all

syms tau_a t T qs Ca0 Ca1 Ca2 Ca3 tau_b qg Cb0 Cb1 Cb2 Cb3 qm vm S delta q_m_x L real;

%% SECOND CUBIC OF THE SPLINE %%
% ti is assumed zero here
tau_a = (2 * t)/T;

% q_A(t)
q_A = qs + Ca0 + Ca1 * tau_a + Ca2 * tau_a^2 +  Ca3 * tau_a^3;

% q_A_dot(t)
q_A_dot = simplify(diff(q_A, t), Steps=1000);

% q_A_dot_dot(t)
q_A_dot_dot = simplify(diff(q_A_dot, t), Steps=1000);



%% SECOND CUBIC OF THE SPLINE %%
tau_b = (2 * t/T) - 1;

% using tau_b - 1 since we are using qg instead of qint
% q_B(t)
q_B = qg + Cb0 + Cb1 * (tau_b - 1) + Cb2 * (tau_b - 1)^2 + Cb3 * (tau_b - 1)^3;

% q_B_dot(t)
q_B_dot = simplify(diff(q_B, t), Steps=1000);

% q_B_dot_dot(t)
q_B_dot_dot = simplify(diff(q_B_dot, t), Steps=1000);



%%   Ca0 Ca1 Ca2 Ca3 Cb0 Cb1 Cb2 Cb3
A = [1,  0,  0,  0,  0,  0,  0,  0;                                                                %q_A(0)
     1,  1 , 1,  1,  0,  0,  0,  0;                                                                %q_A(1)
     0, (2)/T, 0, 0, 0,  0,  0,  0;                                                                %q_A_dot(0)
     0, (2/T^3) * (1 * T^2), (2/T^3) * (2 * 1 * T^2), (2/T^3) * (3 * 1 * T^2), 0, 0, 0, 0;         %q_A_dot(1)
     0,  0,  0,  0,  1, -1, 1, -1;                                                                 %q_B(0) 
     0,  0,  0,  0,  1,  0,  0,  0;                                                                %q_B(1)
     0,  0,  0,  0,  0, (2/T), -(4/T), (6/T);                                                      %q_B_dot(0)
     0,  0,  0,  0,  0, (2/T), 0, 0;                                                               %q_B_dot(1)
    ];

B = [0;       %q_A(0)
    qm - qs;  %q_A(1)
     0;       %q_dot_A(0)
     vm;      %q_dot_A(1)
     qm - qg; %q_B(0)
     0;       %q_B(1)
     vm;      %q_B_dot(0)
     0;       %q_B_dot(1)
    ];

A_B = simplify(A\B, steps=100);

Ca0_n = A_B(1)
Ca1_n = A_B(2)
Ca2_n = A_B(3)
Ca3_n = A_B(4)
Cb0_n = A_B(5)
Cb1_n = A_B(6)
Cb2_n = A_B(7)
Cb3_n = A_B(8)

% q_A(t) with Ca0, Ca1, Ca2, Ca3 substituted 
q_A_subs = simplify(subs(q_A, [Ca0, Ca1, Ca2, Ca3], [Ca0_n, Ca1_n, Ca2_n, Ca3_n]), steps = 100)

% q_A_dot(t) with Ca0, Ca1, Ca2, Ca3 substituted
q_A_dot_subs = simplify(subs(q_A_dot, [Ca0, Ca1, Ca2, Ca3], [Ca0_n, Ca1_n, Ca2_n, Ca3_n]), steps = 100);

% q_A_dot_dot(t) with Ca0, Ca1, Ca2, Ca3 substituted
q_A_dot_dot_subs = simplify(subs(q_A_dot_dot, [Ca0, Ca1, Ca2, Ca3], [Ca0_n, Ca1_n, Ca2_n, Ca3_n]), steps = 100);

% q_B(t) with Cb0, Cb1, Cb2, Cb3 substituted 
q_B_subs = simplify(subs(q_B, [Cb0, Cb1, Cb2, Cb3], [Cb0_n, Cb1_n, Cb2_n, Cb3_n]), steps = 100)

% q_B_dot(t) with Cb0, Cb1, Cb2, Cb3 substituted
q_B_dot_subs = simplify(subs(q_B_dot, [Cb0, Cb1, Cb2, Cb3], [Cb0_n, Cb1_n, Cb2_n, Cb3_n]), steps = 100);

% q_B_dot_dot(t) with Cb0, Cb1, Cb2, Cb3 substituted
q_B_dot_dot_subs = simplify(subs(q_B_dot_dot, [Cb0, Cb1, Cb2, Cb3], [Cb0_n, Cb1_n, Cb2_n, Cb3_n]), steps = 100);



%% Calculate vm
% To calculate vm, we impose the following condition on the acceleration
% q_A_dot_dot(1) = q_B_dot_dot(0) 
% this is the continuity of the acceleration condition

% for cubic A, the tf = T/2
q_A_dot_dot_tf = subs(q_A_dot_dot_subs, t, T/2)
% for cubic B, the ti = T/2
q_B_dot_dot_ti = subs(q_B_dot_dot_subs, t, T/2)

vm_value = simplify(solve(q_A_dot_dot_tf == q_B_dot_dot_ti, vm), steps = 1000)


%% calculate qs, qm and qg 
%% code for the inverse kinematics of a planar PR

syms S q1 px py q1 q2

%% this holds if the robot has the first angle (the fixed one, associated to the prismatic joint) set to 0. An analogous calculation can be done for alpha = pi/2, alpha = pi, alpha = -pi/2

% calculate qs from ps

%% INPUT, CHANGE BASED ON THE REQUEST
px = S;
py = L;

% find q2

s2 = py/L;
c2_pos = + sqrt(1 - s2);
c2_neg = - sqrt(1 - s2);

q2_pos = atan2(s2, c2_pos);
q2_neg = atan2(s2, c2_neg);


% find q1

q1_pos = px - L*c2_pos;
q1_neg = px - L*c2_neg;



%% display the solution

q_first = [q1_pos; q2_pos;];
q_second = [q1_neg; q2_neg;];


qs_value = q_first



% calculate qm from ps

%% INPUT, CHANGE BASED ON THE REQUEST
px = S + (delta/2);
py = L/4;

% find q2

s2 = py/L;
c2_pos = + sqrt(1 - s2);
c2_neg = - sqrt(1 - s2);

q2_pos = atan2(s2, c2_pos);
q2_neg = atan2(s2, c2_neg);


% find q1

q1_pos = px - L*c2_pos;
q1_neg = px - L*c2_neg;



%% display the solution

q_first = [q1_pos; q2_pos;];
q_second = [q1_neg; q2_neg;];


qm_value = q_first



% calculate qg from ps

%% INPUT, CHANGE BASED ON THE REQUEST
px = S + delta;
py = L;

% find q2

s2 = py/L;
c2_pos = + sqrt(1 - s2);
c2_neg = - sqrt(1 - s2);

q2_pos = atan2(s2, c2_pos);
q2_neg = atan2(s2, c2_neg);


% find q1

q1_pos = px - L*c2_pos;
q1_neg = px - L*c2_neg;



%% display the solution

q_first = [q1_pos; q2_pos;];
q_second = [q1_neg; q2_neg;];


qg_value = q_first


vm_q1_value_subs = subs(vm_value, [qg, qs, qm], [qg_value(1), qs_value(1), qm_value(1)])
vm_q2_value_subs = subs(vm_value, [qg, qs, qm], [qg_value(2), qs_value(2), qm_value(2)])

q_A_q1 = subs(q_A_subs, [qg, qs, qm, vm], [qg_value(1), qs_value(1), qm_value(1), vm_q1_value_subs])
q_A_q2 = subs(q_A_subs, [qg, qs, qm, vm], [qg_value(2), qs_value(2), qm_value(2), vm_q2_value_subs])

q_B_q1 = subs(q_B_subs, [qg, qs, qm, vm], [qg_value(1), qs_value(1), qm_value(1), vm_q1_value_subs])
q_B_q2 = subs(q_B_subs, [qg, qs, qm, vm], [qg_value(2), qs_value(2), qm_value(2), vm_q2_value_subs])


Ca2_n_new_q1 = subs(Ca2_n, [qm, qs, vm, qg], [qm_value(1), qs_value(1), vm_q1_value_subs, qg_value(1)])
Ca2_n_new_q2 = subs(Ca2_n, [qm, qs, vm, qg], [qm_value(2), qs_value(2), vm_q2_value_subs, qg_value(2)])
Ca2_n_new_new = subs(Ca2_n_new_q1, [L, S, delta], [1, 0, 3])
Ca2_n_new_new = subs(Ca2_n_new_q2, [L, S, delta], [1, 0, 3])



%% PLOT THE RESULTS


%% Plot for joint q1
A_plot_for_q1 = simplify(subs(q_A_q1, [L, S, delta], [1, 0, 3]), steps = 1000)
B_plot_for_q1 = simplify(subs(q_B_q1, [L, S, delta], [1, 0, 3]), steps = 1000)

T_val = 2;
time = 0: 0.01: T_val;
T_val_2 = 4
time_2 = T_val: 0.01: T_val_2; 

%% DEVI USARE T_VAL 2 PER ENTRAMBI!!!
A_values_for_q1 = subs(A_plot_for_q1, [{t} {T}], [{time} {T_val_2}])
B_values_for_q1 = subs(B_plot_for_q1, [{t} {T}], [{time_2} {T_val_2}])

figure;
plot([time, time_2], [A_values_for_q1, B_values_for_q1])
hold on;
grid on;
xlabel("time");
ylabel("$q_{d,1}(t)$", 'Interpreter', 'latex')

%% Plot for joint q2
A_plot_for_q2 = simplify(subs(q_A_q2, [L, S, delta], [1, 0, 3]), steps = 1000)
B_plot_for_q2 = simplify(subs(q_B_q2, [L, S, delta], [1, 0, 3]), steps = 1000)


T_val = 2;
time = 0: 0.01: T_val
T_val_2 = 4;
time_2 = T_val: 0.01: T_val_2

%% DEVI USARE T_VAL 2 PER ENTRAMBI!!!
A_values_for_q2 = subs(A_plot_for_q2 , [{t} {T}], [{time} {T_val_2}])
B_values_for_q2 = subs(B_plot_for_q2, [{t} {T}], [{time_2} {T_val_2}])

figure;
plot([time, time_2], [A_values_for_q2, B_values_for_q2])
hold on;
grid on;
xlabel("time");
ylabel("$q_{d,2}(t)$", 'Interpreter', 'latex')