function [flattened_vector, sizes] = flatten_cell(input_cell)
    sizes = zeros(length(input_cell), 1);
    for i = 1 : length(input_cell)
        sizes(i) = size(input_cell{i}, 2);
        
        for j = 1 : sizes(i)
            if i == 1
                flattened_vector(j) = input_cell{i}(j);
            else
                flattened_vector(sum(sizes(1:i - 1)) + j) = input_cell{i}(j);
            end
        end
    end
end