function [used_params] = check_sym_appearence(expr, params)
    symbolsInExpr = symvar(expr);
    isPresent = false(size(params));

    for i = 1:size(params, 1)
        for j = 1:size(params, 2)
            % Check if the symbolic variable is in the list of symbolic variables in the expression
            isPresent(i, j) = any(arrayfun(@(s) isequal(s, params(i, j)), symbolsInExpr));
        end
    end

    used_count = 0;
    for i = 1:size(params, 1)
        for j = 1:size(params, 2)
            if isPresent(i, j)
                used_count = used_count + 1;
                used_params_all(used_count) = params(i, j);
            end
        end
    end

    for i = 1:size(params, 1)
        used_params{i} = intersect(params(i, :), used_params_all);
    end
end