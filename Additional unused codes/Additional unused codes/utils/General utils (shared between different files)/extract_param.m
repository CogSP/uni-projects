function param = extract_param(params, key)
    % params is the result of the concatenation of known and unkown params
    for i = 1 : length(params)
        param(i, :) = params{i}(key);
end