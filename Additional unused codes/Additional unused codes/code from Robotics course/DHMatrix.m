function [T, A] = DHMatrix(DHTable)
% -DHTable: a n-vector of vectors composed like this: [alpha a d theta]
    T = eye(4);
    nums = size(DHTable);
    
    A = cell(1,nums(1));
    
    for i = 1:nums(1)
        line = DHTable(i, :);
        R = [cos(line(4)) -cos(line(1))*sin(line(4)) sin(line(1))*sin(line(4)) line(2)*cos(line(4));
             sin(line(4)) cos(line(1))*cos(line(4)) -sin(line(1))*cos(line(4)) line(2)*sin(line(4));
             0 sin(line(1)) cos(line(1)) line(3);
             0 0 0 1;];
        A{i} = R;
        T = T * R;   
    end

    if isa(T, 'sym')
        T = simplify(T);
    end
end