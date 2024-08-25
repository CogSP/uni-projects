function [X, R] = X_matrix(A, i)
    R = A{i}(1:3, 1:3);
    r = A{i}(1:3, 4);
    
    X = [
        R, zeros(3,3);
        skew(r) * R, R;
    ];
end