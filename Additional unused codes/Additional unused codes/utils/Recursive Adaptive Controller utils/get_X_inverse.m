function [X_inverse] = get_X_inverse(X)
    X_inverse = [
        X(1:3, 1:3)', X(1:3, 4:6)';
        X(4:6, 1:3)' , X(4:6, 4:6)';
    ];
end