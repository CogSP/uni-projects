function determinants = calculateMinorsDeterminants(matrix, k)
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
    for i = 1:size(combinationsr, 1)
        % Extract rows and columns based on the combinations
        for j = 1:size(combinationsc, 1)
            minor = matrix(combinationsr(i, :), combinationsc(j, :));
            determinants{i+j*rows} = det(minor);
        end
    end
end
