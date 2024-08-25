function [F, tau, a_hat_dot, D_hat_dot_s, D_hat_dot_v] = AC_Downward(q_dot_r, s, v, omega, omega_dot, e, A, all_params, unknown_params, unknown_params_symbols, D_hat_s, D_hat_v, gain_parameters)
    %Initialization
    P = gain_parameters.P;
    P_s = gain_parameters.P_s;
    P_v = gain_parameters.P_v;
    K_d = gain_parameters.K_d;

    dof = length(q_dot_r);
    F = cell(dof, 1);
    tau = zeros(dof, 1);

    
    a_hat_dot = cell(dof, 1);
    D_hat_dot_s = zeros(1, dof);
    D_hat_dot_v = zeros(1, dof);

    %Iterations

    % params = sym('a%d', [1, 10], 'real');

    for i = dof: -1 : 1
        [X, rotation] = X_matrix(A,i);
        X_star = get_X_star(X); % Not sure about this!

        z = rotation' * [0; 0; 1;]; %must be checked
        d = [z; zeros(3,1)];

        [B_i, I_bar] = create_B(v, i, all_params{i}.keys);
        
        aux_sum = I_bar * omega_dot(:, i) + B_i * omega(:, i);

        regressor = jacobian(aux_sum, unknown_params_symbols{i});
        f_i = subs(regressor, unknown_params_symbols{i}, unknown_params{i}');
        aux_sum = double(subs(aux_sum, all_params{i}.keys, all_params{i}.values));

        if i == dof
            F{i} = aux_sum;
        else
            F{i} = aux_sum + X_star * F{i + 1};
        end

        % F{i}
        tau(i) = d' * F{i} - K_d(i, i) * s(i) + D_hat_s(i) * sign(q_dot_r(i)) + D_hat_v(i) * q_dot_r(i);
        
        %ADAPTATION
        a_hat_dot{i} = -P(i, i) * e(:, i)' * f_i;

        D_hat_dot_s(i) = -P_s(i, i) * s(i) * sign(q_dot_r(i));
        D_hat_dot_v(i) = -P_v(i, i) * s(i) * q_dot_r(i);
    end
end