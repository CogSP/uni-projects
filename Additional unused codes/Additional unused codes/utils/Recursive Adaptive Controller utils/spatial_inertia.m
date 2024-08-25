function I_bar = spatial_inertia(parameters)
    % ugly assignment of parameters for a single link
    J_x_x = parameters(1);
    J_x_y = parameters(2);
    J_x_z = parameters(3);
    J_y_y = parameters(4);
    J_y_z = parameters(5);
    J_z_z = parameters(6);
    p_x = parameters(7);
    p_y = parameters(8);
    p_z = parameters(9);
    m = parameters(10);

    % Define the inertia matrix I
    I = [
        J_x_x, J_x_y, J_x_z;
        J_x_y, J_y_y, J_y_z;
        J_x_z, J_y_z, J_z_z;
    ];

    % Define the skew-symmetric matrix p
    p = skew([p_x; p_y; p_z]);

    % Define the spatial inertia matrix I_bar
    I_bar = [
        I, p;
        -p, m * eye(3)
    ];
end
