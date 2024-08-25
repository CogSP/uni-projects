function [unknown_used_params, unknown_used_params_symbols, known_used_params] = divide_used_params(used_params, unknown_params_symbols, unknown_params, known_params)
    unknown_used_params = cell(length(used_params), 1);
    unknown_used_params_symbols = cell(length(used_params), 1);
    known_used_params = cell(length(used_params), 1);

    for i = 1:length(used_params)
        known_count = 0;
        unknown_count = 0;
        for j = 1:length(used_params{i})
            if isKey(known_params{i}, char(used_params{i}(j)))
                known_count = known_count + 1;

                param_names(known_count) = used_params{i}(j);
                param_values(known_count) = known_params{i}(param_names);
            else
                unknown_count = unknown_count + 1;
                unknown_used_params_symbols{i}(unknown_count) = used_params{i}(j);

                index = unknown_params_symbols{i} == used_params{i}(j);
                % index = find(unknown_params_symbols{i} == used_params{i}(j))
                unknown_used_params{i}(unknown_count) = unknown_params{i}(index);
            end
        end
        
        if known_count > 0
            params = dictionary(param_names, param_values);
            known_used_params{i} = params;
        else
            params = dictionary([], []);
            known_used_params{i} = params;
        end
    end
end