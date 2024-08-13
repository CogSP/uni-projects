syms q1 q2 q3 q4 a b
 
J = [-q3*sin(q2) - b*sin(q2 + q4), -q3*sin(q2)-b*sin(q2 + q4), cos(q2), -b*sin(q2 + q4);
     a + q3*cos(q2) + b*cos(q2 + q4), q3*cos(q2) + b*cos(q2 + q4), sin(q2), b*cos(q2 + q4);
     1, 1, 0, 1]

matrix = J
k = 3

% Check if the matrix is square
[rows, cols] = size(matrix);

% Check if k is within the matrix dimensions
if k < 1 || k > min(rows,cols)
    error('Invalid value for k.');
end

determinants = cell(nchoosek(rows, k) * nchoosek(cols, k), 1);

% Generate all combinations of rows and columns to form minors of order k
combinationsr = nchoosek(1:rows, k);
combinationsc = nchoosek(1:cols, k);
k=0
for i = 1:size(combinationsr, 1)
    % Extract rows and columns based on the combinations
    for j = 1:size(combinationsc, 1)
        k=k+1
        disp("i:")
        disp(i)
        disp("j:")
        disp(j)
        minor = matrix(combinationsr(i, :), combinationsc(j, :))
        determinants{k} = simplify(det(minor), steps=100)
    end
end

