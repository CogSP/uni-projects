function [omega, omega_dot, a_c] = NE_forward(q, q_dot, q_ddot, A, g0, r_c, r)
    % Initialize variables 
    num_links = length(q);
    z = [0; 0; 1];
    omega_0 = zeros(3, 1);
    omega_dot_0 = zeros(3, 1);
    
    % Preallocate arrays
    omega = cell(num_links, 1);     % Preallocating a 1xN cell array for omega
    omega_dot = cell(num_links, 1); % Preallocating a 1xN cell array for omega_dot
    a = cell(num_links, 1);       % Preallocating a 1xN cell array for a
    a_c = cell(num_links, 1);       % Preallocating a 1xN cell array for a_c

    % Initial conditions
    omega{1} = A{1}(1:3, 1:3)' * (omega_0 + q_dot(1) * z); %generalized!
    omega_dot{1} = A{1}(1:3, 1:3)' * (omega_dot_0 + q_ddot(1) * z + q_dot(1) * cross(omega_0, z)); %generalized!

    a{1} = A{1}(1:3, 1:3)' * (-g0) + cross(omega_dot{1}, r{1}) + cross(omega{1}, cross(omega{1}, r{1})); %generalized!

    a_c{1} = a{1} + cross(omega_dot{1}, r_c(:, 1)) + cross(omega{1}, cross(omega{1}, r_c(:, 1)));

    % Compute for each link
    for i = 2:num_links
        omega{i} = A{i}(1:3, 1:3)' * (omega{i-1} + q_dot(i) * z);
        omega_dot{i} = A{i}(1:3, 1:3)' * (omega_dot{i-1} + q_ddot(i) * z + q_dot(i) * cross(omega{i-1}, z));

        a{i} = A{i}(1:3, 1:3)' * a{i-1} + cross(omega_dot{i}, r{i}) + cross(omega{i}, cross(omega{i}, r{i}));
        a_c{i} = a{i} + cross(omega_dot{i}, r_c(:, i)) + cross(omega{i}, cross(omega{i}, r_c(:, i)));
    end
end