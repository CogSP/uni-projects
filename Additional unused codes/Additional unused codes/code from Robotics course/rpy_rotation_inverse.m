function [phi, theta, psi] = rpy_rotation_inverse(R, sequence)
    sequence = char(sequence);
    [phi, theta, psi] = euler_rotation_inverse(R, flip(sequence));
end