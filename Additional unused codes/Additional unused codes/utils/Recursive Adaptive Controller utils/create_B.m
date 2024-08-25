function [B, I_bar] = create_B(v, i, all_params)
    J_x_x = sym(sprintf('J_x_x_%d', i), 'real');
    J_x_y = sym(sprintf('J_x_y_%d', i), 'real');
    J_x_z = sym(sprintf('J_x_z_%d', i), 'real');
    J_y_y = sym(sprintf('J_y_y_%d', i), 'real');
    J_y_z = sym(sprintf('J_y_z_%d', i), 'real');
    J_z_z = sym(sprintf('J_z_z_%d', i), 'real');
    p_x = sym(sprintf('p_x_%d', i), 'real');
    p_y = sym(sprintf('p_y_%d', i), 'real');
    p_z = sym(sprintf('p_z_%d', i), 'real');
    m = sym(sprintf('m%d', i), 'real');

    I_bar = spatial_inertia([J_x_x J_x_y J_x_z J_y_y J_y_z J_z_z p_x p_y p_z m]);

    % List of all symbolic variables in I_bar
    all_symbols = symvar(I_bar);

    % Symbols to be removed (those not in all_params)
    symbols_to_remove = setdiff(all_symbols, all_params);

    % Substitute the symbols not in all_params with zero
    I_bar = subs(I_bar, symbols_to_remove, zeros(size(symbols_to_remove)));

    I = [
        J_x_x, J_x_y, J_x_z;
        J_x_y, J_y_y, J_y_z;
        J_x_z, J_y_z, J_z_z;
    ];
    mc = [
        p_x;
        p_y;
        p_z;
    ];
    mass = m;
    angular_v = v(1:3, i);
    temp = [
        -skew(I * angular_v), skew(mc) * skew(angular_v);
        -skew(angular_v) * skew(mc), mass * skew(angular_v);
    ];
    B = -I_bar * spatial_skew(v(:, i)) + temp;

    % List of all symbolic variables in B
    all_symbols = symvar(B);

    % Symbols to be removed (those not in all_params)
    symbols_to_remove = setdiff(all_symbols, all_params);

    % Substitute the symbols not in all_params with zero
    B = subs(B, symbols_to_remove, zeros(size(symbols_to_remove)));
end