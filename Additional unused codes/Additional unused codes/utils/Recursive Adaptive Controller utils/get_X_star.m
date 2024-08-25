function [X_star] = get_X_star(X)
    %I modified this! coming back!
    % X_star = [
    %     X(1:3, 1:3)', X(1:3, 4:6)';
    %     X(4:6, 1:3)' , X(4:6, 4:6)';
    % ];
    % 
    % X_star = inv(X_star);
    X_star = X';
end