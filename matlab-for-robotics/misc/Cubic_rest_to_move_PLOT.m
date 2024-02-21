syms vf qi qf Ca0 Ca1 Ca2 Ca3 t ti tf T

%% IMPORTANT NOTE: Ca0, Ca1, ... are delta_Q * a, delta_Q * b, ... respectively

assume(t, 'positive')
assume(ti, 'positive')
assume(tf, 'positive')
assume(T, 'positive')

%% CHANGE THESE TO PLOT CORRECTLY THE FUNCTIONS
%% LEAVE THEM COMMENTED IF YOU WANT THE FUNCTIONS EXPRESSED IN SYMBOLS 
%ti = 1.5
%qi = 0
%qf = pi
%vf = 3
% tf is not substituted since we use T and we substitute T afterwards 

% condition
% initial point q(ti)
q_t = qi

% final point
q_T = qf

% starts at rest
q_dot_t = 0

% finishes with a velocity different from 0
q_dot_T = vf

%% THIS MAY CHANGE, E.G. IF YOU HAVE 2 SPLINE
% T = tf - ti
tau = (t - ti)/T 

% cubic q(t)
q = qi + Ca0 + Ca1 * tau + Ca2 * tau^2 + Ca3 * tau^3

% q(ti) is just qi

% q(tf)
q_subs_qf = simplify(subs(q, t, tf), steps = 100) 


% q_dot(t)
q_dot = simplify(diff(q, t), steps = 100)

% q_dot(ti)
q_dot_subs_qi = simplify(subs(q_dot, t, ti), steps = 100)

% q_dot(tf)
q_dot_subs_qf = simplify(subs(q_dot, t, tf), steps = 100)


% q_dot_dot(t)
q_dot_dot = simplify(diff(q_dot, t), steps=1000)

% q_dot_dot(ti)
q_dot_dot_subs_qi = simplify(subs(q_dot_dot, t, ti), steps = 100)

% q_dot_dot(tf)
q_dot_dot_subs_qf = simplify(subs(q_dot_dot, t, tf), steps = 100)


%   Ca0 Ca1 Ca2 Ca3
A = [1, 0,  0,  0,; % q(0)   
     1, 1 , 1,  1;  % q(1)
     0, 1/T, 0, 0 % q_dot(0)
     0, 1/T, 2/T, 3/T]  %q_dot(1)

B = [0;     % q(0)
     qf-qi; % q(1)
     0;     % q_dot(0)
     vf]    % q_dot(1)

A_B = simplify(A\B, steps = 100)

Ca0_n = A_B(1)
Ca1_n = A_B(2)
Ca2_n = A_B(3)
Ca3_n = A_B(4)

% q(t) with Ca0, Ca1, Ca2, Ca3 substituted 
q_subs = simplify(subs(q, [Ca0, Ca1, Ca2, Ca3], [Ca0_n, Ca1_n, Ca2_n, Ca3_n]), steps = 100)

% q_dot(t) with Ca0, Ca1, Ca2, Ca3 substituted
q_dot_subs = simplify(subs(q_dot, [Ca0, Ca1, Ca2, Ca3], [Ca0_n, Ca1_n, Ca2_n, Ca3_n]), steps = 100)

% q_dot_dot(t) with Ca0, Ca1, Ca2, Ca3 substituted
q_dot_dot_subs = simplify(subs(q_dot_dot, [Ca0, Ca1, Ca2, Ca3], [Ca0_n, Ca1_n, Ca2_n, Ca3_n]), steps = 100)

%% PLOTTING
% T_val is ti - tf
T_val = 2; 
time = 0: 0.01: T_val;
q_values = subs(q_subs, [{t} {T}], [{time} {T_val}])
q_dot_values = subs(q_dot_subs, [{t} {T}], [{time} {T_val}])
q_dot_dot_values = subs(q_dot_dot_subs, [{t} {T}], [{time} {T_val}])

figure;
plot(time, q_values)
hold on;
grid on;
xlabel("time");
ylabel("q(t)")

figure;
plot(time, q_dot_values)
hold on;
grid on;
xlabel("time");
ylabel("$\dot{q}(t)$", 'Interpreter', 'latex')


figure;
plot(time, q_dot_dot_values)
hold on;
grid on;
xlabel("time");
ylabel("$\ddot{q}(t)$", 'Interpreter', 'latex')
