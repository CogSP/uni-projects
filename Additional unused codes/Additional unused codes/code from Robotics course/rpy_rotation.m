function R = rpy_rotation(angles, sequence)
    sequence = char(sequence);
    R = euler_rotation(flip(angles), flip(sequence));
end