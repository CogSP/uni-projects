function [all_params, unknown_params] = get_all_parameters(dof, a_adapt, known_params, unknown_params_sizes, unknown_params_symbols)
    all_params = cell(dof, 1);
    unknown_dictionaries = cell(dof, 1);
    unknown_params = cell(dof, 1);
    temp = 0;
    for i = 1 : size(unknown_params_sizes, 1)
        if i == 1
            unknown_params{i} = a_adapt(1:unknown_params_sizes(i));
        else
            start = temp + 1;
            unknown_params{i} = a_adapt(start:start + unknown_params_sizes(i) - 1);
        end
        temp = temp + unknown_params_sizes(i);

        unknown_dictionaries{i} = dictionary(unknown_params_symbols{i}, unknown_params{i}');
        all_params{i} = known_params{i};

        all_keys = [known_params{i}.keys; unknown_params_symbols{i}'];
        all_values = [known_params{i}.values; unknown_params{i}];
        all_params{i} = dictionary(all_keys, all_values);
    end
end