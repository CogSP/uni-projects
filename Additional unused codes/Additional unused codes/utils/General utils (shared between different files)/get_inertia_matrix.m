function I = get_inertia_matrix(params)
    I = cell(size(params, 1), 1);
    for i = 1:size(params, 1)
        I{i} = [
            params(i, 1), params(i, 2), params(i, 3);
            params(i, 2), params(i, 4), params(i, 5);
            params(i, 3), params(i, 5), params(i, 6);
        ];
    end
end