function u = NE_backward(q_dot, omega, omega_dot, a_c, A, m, I, r_c, r, Fv, Fs)
    num_links = length(q_dot);
    z = [0; 0; 1];

    % Initialize variables
    f = cell(num_links, 1);
    tau = cell(num_links, 1);
    
    % Calculate forces and moments for the last link
    f{num_links} = m(num_links) * a_c{num_links};
    tau{num_links} = -cross(f{num_links}, r{num_links} + r_c(:, num_links)) ...
        + I{num_links} * omega_dot{num_links} ...
        + cross(omega{num_links}, I{num_links} * omega{num_links});

    % Compute for the rest of the links
    for i = num_links-1:-1:1
        f{i} = A{i + 1}(1:3, 1:3) * f{i+1} + m(i) * a_c{i};
        tau{i} = A{i + 1}(1:3, 1:3) * tau{i+1} ...
            + cross(A{i + 1}(1:3, 1:3) * f{i+1}, r_c(:, i)) ...
            - cross(f{i}, r{i} + r_c(:, i)) ...
            + I{i} * omega_dot{i} + cross(omega{i}, I{i} * omega{i});
    end

    for i = 1:num_links
        z_i = A{i}(1:3, 1:3)' * z;
        u(i, :) = tau{i}' * z_i +Fv(i)*q_dot(i) + Fs(i)*sign(q_dot(i));
    end

    %u = simplify(u);
end