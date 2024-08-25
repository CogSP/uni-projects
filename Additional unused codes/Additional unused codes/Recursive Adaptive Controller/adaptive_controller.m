function [f, tau, a_hat_dot, D_hat_dot_s, D_hat_dot_v]= adaptive_controller(q, q_dot, q_d, q_dot_d, q_ddot_d, alpha, g0, DHTable, all_params, unknown_params, unknown_params_symbols, D_hat_s, D_hat_v, gains_parameters)
    q_dot_r = q_dot_d - alpha * (q - q_d);
    s = q_dot - q_dot_r;
    q_ddot_r = q_ddot_d - alpha * (q_dot - q_dot_d); %not written 

    [~, A] = DHMatrix(DHTable);
    omega_dot_0 = g0; %I changed from - g0 to g0

    [v, omega, omega_dot, e] = AC_Upward(q_dot, q_dot_r, q_ddot_r, A, omega_dot_0);

    [f, tau, a_hat_dot, D_hat_dot_s, D_hat_dot_v] = AC_Downward(q_dot_r, s, v, omega, omega_dot, e, A , all_params, unknown_params, unknown_params_symbols, D_hat_s, D_hat_v, gains_parameters);
end
