function [theta, r] = angle_axis_rotation_inverse(R)
    s_theta = 1/2 * sqrt((R(1, 2) - R(2, 1))^2 ...
        + (R(1, 3) - R(3, 1))^2 ...
        + (R(2, 3) - R(3, 2))^2);
    s_theta = [s_theta -s_theta];
    c_theta = (trace(R) - 1) / 2;

    if s_theta(1) ~= 0
        theta = atan2(s_theta, c_theta);

        rx = R(3, 2) - R(2, 3);
        ry = R(1, 3) - R(3, 1);
        rz = R(2, 1) - R(1, 2);
        r1 = 1 / (2 * s_theta(1)) * [rx; ry; rz];
        r2 = 1 / (2 * s_theta(2)) * [rx; ry; rz];
        r = [r1 r2];
    elseif c_theta == -1
        theta = [sym(pi) -sym(pi)];

        rx = sqrt((R(1, 1) + 1) / 2);
        rx = [rx -rx];
        ry = sqrt((R(2, 2) + 1) / 2);
        ry = [ry -ry];
        rz = sqrt((R(3, 3) + 1) / 2);
        rz = [rz -rz];

        if R(1, 2) > 0 && R(1, 3) > 0
            r = [rx; ry; rz];
        elseif R(2, 3) > 0
            r = [-rx; ry; rz];
        elseif R(1, 3) > 0
            r = [rx; -ry; rz];
        else
            r = [rx; ry; -rz];
        end
    else
        disp("No Solution");
    end
end