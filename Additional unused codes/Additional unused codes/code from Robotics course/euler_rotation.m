function R = euler_rotation(angles, sequence)
    if strlength(sequence) ~= 3
        disp("Sequence not valid, must be of lenght three.")
        return;
    end
    
    sequence = lower(char(sequence));
    if (sequence(1) == sequence(2) || sequence(2) == sequence(3))
        disp("Two consecutive rotation along the same axis are not valid.")
        return
    end

    R = simple_rotation(angles(1), sequence(1))...
        * simple_rotation(angles(2), sequence(2))...
        * simple_rotation(angles(3), sequence(3));
end