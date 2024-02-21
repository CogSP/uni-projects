clc
clear all

syms r s h C V A L t

assume(r, 'positive')
assume(h, 'positive')
assume(t, 'positive')

p = [r*sin(s); h*s; r*cos(s)] + C

p_prime = diff(p, s)

p_prime_norm = simplify(sqrt(p_prime.' * p_prime), steps=100)

p_prime_prime = diff(p_prime, s)

p_prime_prime_norm = simplify(sqrt(p_prime_prime.' * p_prime_prime), steps=100)

t_s = simplify(p_prime / p_prime_norm, steps=100)

t_s_prime = diff(t_s, s)

t_s_prime_norm = simplify(sqrt(t_s_prime.' * t_s_prime), steps=100)

n_s = simplify(t_s_prime / t_s_prime_norm, steps = 100)

b_s = simplify(cross(t_s, n_s), steps=100)


p_prime_times_t = simplify(p_prime.' * t_s, steps=100)

p_prime_prime_times_t = p_prime_prime.' * t_s

p_prime_times_n = p_prime.' * n_s

p_prime_prime_times_n = simplify(p_prime_prime.' * n_s, steps=100)


v_max = double(subs(min(sqrt(A/r), V/sqrt(r^2 + h^2)), [A, r, V, h], [4.5, 0.4, 2, 0.3]))

a_max = double(subs(A/sqrt(r^2+h^2), [A, r, h], [4.5, 0.4, 0.3]))


T = double(subs((L * a_max + (v_max)^2) / (a_max * v_max), [L], [4*pi]))


T_s = v_max / a_max

function_1 = (a_max*t^2)/2
time_1 = 0: 0.01: T_s

function_2 = v_max*t - (v_max^2)/(2*a_max)
time_2 = T_s: 0.01: (T - T_s)

function_3 = -((a_max*(t - T)^2)/2) + v_max*T - (v_max^2)/a_max
time_3 = (T - T_s): 0.01: T


function_1_values = subs(function_1, [{t}], [{time_1}]);
function_2_values = subs(function_2, [{t}], [{time_2}]);
function_3_values = subs(function_3, [{t}], [{time_3}]);


figure;
plot([time_1, time_2, time_3], [function_1_values, function_2_values, function_3_values])
hold on;
grid on;
xlabel("time");
ylabel("sigma(t)")


function_1_diff = diff(function_1, t)
function_2_diff = diff(function_2, t)
function_3_diff = diff(function_3, t)

function_1_diff_values = subs(function_1_diff, [{t}], [{time_1}]);
function_2_diff_values = subs(function_2_diff, [{t}], [{time_2}]);
function_3_diff_values = subs(function_3_diff, [{t}], [{time_3}]);


figure;
plot([time_1, time_2, time_3], [function_1_diff_values, function_2_diff_values, function_3_diff_values])
hold on;
grid on;
xlabel("time");
ylabel("sigma(t)")


function_1_diff_diff = diff(function_1_diff, t)
function_2_diff_diff = diff(function_2_diff, t)
function_3_diff_diff = diff(function_3_diff, t)

function_1_diff_diff_values = subs(function_1_diff_diff, [{t}], [{time_1}]);
function_2_diff_diff_values = subs(function_2_diff_diff, [{t}], [{time_2}]);
function_3_diff_diff_values = subs(function_3_diff_diff, [{t}], [{time_3}]);


figure;
plot([time_1, time_2, time_3], [function_1_diff_diff_values, function_2_diff_diff_values, function_3_diff_diff_values])
hold on;
grid on;
xlabel("time");
ylabel("sigma(t)")
