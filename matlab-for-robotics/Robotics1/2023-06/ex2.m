syms s tau delta R A t T tau_sym

assume(T, 'positive')

p_s = R * [cos(s);
           sin(s)];

%p_dot_dot = (R * delta)/T * ((-vector_1 * ((-6*tau^2) + (6 * tau))^2) + (vector_2/T * (-12*tau +6)))

tau = t/T

s_t = delta * (-2 * tau^3 + 3 * tau^2)

p_dot_s = simplify(diff(p_s, s) * diff(s_t, t), steps = 1000)
p_dot_dot_s = simplify((diff(diff(p_s, s), s) * diff(s_t, t)^2) + diff(p_s, s) * diff(diff(s_t, t), t) , steps = 1000)


tau_sym = t/T

s_diff_4 = simplify(subs(diff(s_t, t)^4, t, tau_sym*T), steps=1000)
s_diff_diff_2 = simplify(subs(diff(diff(s_t, t), t)^2, t, tau_sym*T), steps=1000)
p_dot_dot_s_norm = simplify(R*sqrt(s_diff_4 + s_diff_diff_2), steps=1000)

%alpha_tau = 36*delta^2*tau_sym^4*(1 - tau_sym)^4 + (1 - 2*tau_sym)^2

%alpha_tau_diff = diff(alpha_tau, tau_sym)

% p_dot_dot_s_norm = simplify(sqrt(p_dot_dot_s.' * p_dot_dot_s), steps = 1000)
% 
%in questi intervalli abbiamo che la norma è massima
% p_dot_dot_s_norm2 = simplify(subs(p_dot_dot_s_norm, t, 0), steps = 1000)
% p_dot_dot_s_norm3 = simplify(subs(p_dot_dot_s_norm, t, T), steps = 1000)
% 
% roots = simplify(solve(diff(p_dot_dot_s_norm , t) == 0, t), steps = 1000)
% 
% root_1 = simplify(subs(p_dot_dot_s_norm, t, T/2 - (((6*T^6)/delta^2)^(1/3)/3 + T^2 - (243^(1/6)*((2*T^6)/delta^2)^(1/3)*1i)/3)^(1/2)/2), steps = 1000)
% root_2 = simplify(subs(p_dot_dot_s_norm, t, T/2 + (((6*T^6)/delta^2)^(1/3)/3 + T^2 - (243^(1/6)*((2*T^6)/delta^2)^(1/3)*1i)/3)^(1/2)/2), steps = 1000)
% root_3 = simplify(subs(p_dot_dot_s_norm, t, T/2 - (3^(1/2)*(3*T^2 + 6^(1/3)*(T^6/delta^2)^(1/3) + 2^(1/3)*3^(5/6)*(T^6/delta^2)^(1/3)*1i)^(1/2))/6), steps = 1000)
% root_4 = simplify(subs(p_dot_dot_s_norm, t, T/2 + (3^(1/2)*(3*T^2 + 6^(1/3)*(T^6/delta^2)^(1/3) + 2^(1/3)*3^(5/6)*(T^6/delta^2)^(1/3)*1i)^(1/2))/6), steps = 1000)
% root_5 = simplify(subs(p_dot_dot_s_norm, t, T/2), steps = 1000)
% root_6 = simplify(subs(p_dot_dot_s_norm, t, T/2 - (T^2 - (2*((6*T^6)/delta^2)^(1/3))/3)^(1/2)/2), steps = 1000)
% root_7 = simplify(subs(p_dot_dot_s_norm, t, T/2 + (T^2 - (2*((6*T^6)/delta^2)^(1/3))/3)^(1/2)/2), steps = 1000)

% T_val = simplify(solve(root_5 == A, T), steps = 1000)
% 
% T_val = simplify(subs(T_val, [A, R, delta], [3, 1.5, pi]), steps= 1000)

T_val = 3.3322
time = 0: 0.01: T_val;
p_dot_dot_s_norm = subs(p_dot_dot_s_norm, [A, R, delta, {t} {T}], [3, 1.5, pi, {time} {T_val}])

figure;
plot(time, p_dot_dot_s_norm)
hold on;
grid on;
xlabel("time");
ylabel("norm of the acceleration")


p_x_dot_dot = subs(R*(-cos(s_t)*diff(s_t, t)^2 - sin(s_t)*diff(diff(s_t, t), t)), [A, R, delta, {t} {T}], [3, 1.5, pi, {time} {T_val}])


figure;
plot(time, p_x_dot_dot)
hold on;
grid on;
xlabel("time");
ylabel("x component of the acceleration")


p_y_dot_dot = subs(R*(-sin(s_t)*diff(s_t, t)^2 + cos(s_t)*diff(diff(s_t, t), t)), [A, R, delta, {t} {T}], [3, 1.5, pi, {time} {T_val}])


figure;
plot(time, p_y_dot_dot)
hold on;
grid on;
xlabel("time");
ylabel("y component of the acceleration")


