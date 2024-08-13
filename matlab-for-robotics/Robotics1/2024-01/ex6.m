clc 
clear all

syms vf qi qf Ca1 Ca2 Ca3 t ti tf

assume(t, 'positive')
assume(ti, 'positive')
assume(tf, 'positive')
 
%condizioni
%punto iniziale
q_t = qi

%punto finale
q_T = qf

%inizia a riposo
q_dot_t = 0

%arriva ad una velocità diversa di 0
q_dot_T = vf

tau = (t - ti)/(tf - ti) 

%cubica
q = qi + Ca1 * tau + Ca2 * tau^2 + Ca3 * tau^3

q_dot = simplify(diff(q, t), steps = 100)

q_subs_qf = simplify(subs(q, t, tf), steps = 100) 

q_dot_subs_qi = simplify(subs(q_dot, t, ti), steps = 100)

q_dot_subs_qf= simplify(subs(q_dot, t, tf), steps = 100)

A = [1 , 1, 1; 
      1/(tf -ti), 0, 0
      1/(tf -ti) , 2/(tf -ti), 3/(tf -ti)]  

b=[qf-qi;  %qi va messo a destra, lo avevamo nell'eq
   0;
   vf]

A_B = simplify(A\b, steps = 100)

Ca1_n = A_B(1)
Ca2_n = A_B(2)
Ca3_n = A_B(3)

q_subs = simplify(subs(q, [Ca1, Ca2, Ca3], [Ca1_n, Ca2_n, Ca3_n]), steps = 100)

q_dot_subs = simplify(subs(q_dot, [Ca1, Ca2, Ca3], [Ca1_n, Ca2_n, Ca3_n]), steps = 100)

%domanda 2
q_dot_dot = simplify(diff(q_dot_subs, t), steps = 100)
 
max_t = simplify(solve(q_dot_dot == 0, t), steps = 100)

norm_q_dot_subs_t = simplify(subs(q_dot_subs, t, max_t(1)), steps = 100)

%domanda 3
V_max = double(subs(norm_q_dot_subs_t, [ti, tf, qi, qf, vf], [1.5, 2, pi/2, pi, -4]))

time = double(subs(max_t, [ti, tf, qi, qf, vf], [1.5, 2, pi/2, pi, -4]))

tau = (time - 1.5)/(2-1.5)
