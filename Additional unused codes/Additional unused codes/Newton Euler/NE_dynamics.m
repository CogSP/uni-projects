function tau = NE_dynamics(q, q_dot, q_ddot, g0, DHTable, m, I, r_CoM, Fv, Fs)
    % Compute DH parameters and transformation matrices
    r = cell(length(q), 1);

    for i = 1:length(q) %to obtain the "r vectors"
        row = DHTable(i, :);
        alpha = row(1);
        a = row(2);
        d = row(3);
        r{i} = create_r(d,a,alpha);
    end

    [~, A] = DHMatrix(DHTable);

    % Perform forward recursion
    [omega, omega_dot, a_c] = NE_forward(q, q_dot, q_ddot, A, g0, r_CoM, r);

    % Perform backward recursion
    tau = NE_backward(q_dot, omega, omega_dot, a_c, A, m, I, r_CoM, r, Fv, Fs);
end