function R = angle_axis_rotation(theta, r)
    c = (1 - cos(theta));
    R = [
        r(1)^2*c+cos(theta) r(1)*r(2)*c-r(3)*sin(theta) r(1)*r(3)*c+r(2)*sin(theta);
        r(1)*r(2)*c+r(3)*sin(theta) r(2)^2*c+cos(theta) r(2)*r(3)*c-r(1)*sin(theta);
        r(1)*r(3)*c-r(2)*sin(theta) r(2)*r(3)*c+r(1)*sin(theta) r(3)^2*c+cos(theta)
    ];
end