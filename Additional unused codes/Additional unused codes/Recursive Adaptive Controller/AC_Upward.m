function [v, omega, omega_dot, e] = AC_Upward(q_dot, q_dot_r, q_ddot_r, A, omega_dot_0)
    %INITIALIZATION
    dof = length(q_dot);
    
    v = zeros(6, dof);
    omega = zeros(6, dof);
    omega_dot = zeros(6, dof);
    e = zeros(6, dof);
    
    z = [0; 0; 1];
    for i = 1: dof
        [X, ~] = X_matrix(A, i);
        X_inv = get_X_inverse(X);
        % z = rotation' * [0; 0; 1];
        d = [z; zeros(3,1)];

        if i == 1
            % v(:, i) = d * q_dot(i);
            v(:, i) = X_inv * d * q_dot(i);
            % omega(:, i) = d * q_dot_r(i);
            omega(:, i) = X_inv * d * q_dot_r(i);
            % omega_dot(:, i) = X_inv * omega_dot_0 + d * q_ddot_r(i) + spatial_cross_product(v(:, i), d * q_dot_r(i));
            omega_dot(:, i) = X_inv * (omega_dot_0 + d * q_ddot_r(i));
        else
            % v(:, i) = X_inv * v(:, i-1) + d * q_dot(i);
            v(:, i) = X_inv * (v(:, i-1) + d * q_dot(i));
            % omega(:, i) = X_inv * omega(:, i-1) + d * q_dot_r(i);
            omega(:, i) = X_inv * (omega(:, i-1) + d * q_dot_r(i));
            % omega_dot(:, i) = X_inv * omega_dot(:, i-1) + d * q_ddot_r(i) + spatial_cross_product(v(:, i), d * q_dot_r(i));
            omega_dot(:, i) = X_inv * (omega_dot(:, i-1) + d * q_ddot_r(i) + spatial_cross_product(v(:, i-1), d * q_dot_r(i)));
        end

        e(:, i) = v(:, i) - omega(:, i);
    end

end
