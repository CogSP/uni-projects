function dstate_dt = dynamic_parameter_system(t, state, dof, known_params, unknown_params_sizes, unknown_params_symbols, q_d_t, q_dot_d_t, q_ddot_d_t, symbolic_acceleration, alpha, g0, DHTable, gains_parameters, time, theta, theta_dot, u) %theta added for the subs
    q = state(1:dof);
    q_dot = state(dof + 1 : 2 * dof);
    param_size = sum(unknown_params_sizes, 1);
    a_adapt = state(2 * dof + 1 : 2 * dof + param_size);

    [all_params, unknown_params] = get_all_parameters(dof, a_adapt, known_params, unknown_params_sizes, unknown_params_symbols); %to get everything as needed

    D_hat_s = state(2 * dof + param_size + 1 : 2 * dof + param_size + dof);
    % D_hat_s = reshape(D_hat_s, dof, dof);
    D_hat_v = state(2 * dof + param_size + dof + 1 : 2 * dof + param_size + 2 * dof);
    % D_hat_v = reshape(D_hat_v, dof, dof);

    subs_DHTable = subs(DHTable, theta, q);

    % Interpolate desired values at time t
    q_d = double(subs(q_d_t, time, t));
    q_dot_d = double(subs(q_dot_d_t, time, t));
    q_ddot_d = double(subs(q_ddot_d_t, time, t));
    [~, tau, a_hat_dot, D_hat_dot_s, D_hat_dot_v] = adaptive_controller(q, q_dot, q_d, q_dot_d, q_ddot_d, alpha, [zeros(3, 1); g0], subs_DHTable, all_params, unknown_params, unknown_params_symbols, D_hat_s, D_hat_v, gains_parameters);
    dq_dt = state(dof + 1 : 2 * dof);
    da_dt = a_hat_dot; %length of unknown parameters
    dD_hat_s_dt = double(subs(D_hat_dot_s));
    dD_hat_v_dt = double(subs(D_hat_dot_v));
    dq_dt2 = double(subs(symbolic_acceleration, [theta, theta_dot, u], [q, q_dot, tau]));

    [flattened_da_dt, ~] = flatten_cell(da_dt);

    dstate_dt = double([dq_dt(:); dq_dt2(:); flattened_da_dt'; dD_hat_s_dt(:); dD_hat_v_dt(:)]);
end