function fr = get_fr(T)
    if ~isa(T, 'sym')
        disp("We need a sym homogeneous matrix.");
        fr = -1;
        return
    end
    
    alpha_z = sum(symvar(T(1:3, 1:3)));
    fr = [T(1, 4); T(2, 4); T(3, 4); alpha_z];
end