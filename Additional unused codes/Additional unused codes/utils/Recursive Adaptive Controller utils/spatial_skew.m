function [matrix] = spatial_skew(vector)
    skew_first = skew(vector(1:3));
    skew_second = skew(vector(4:6));

    matrix = [
        skew_first, zeros(3, 3);
        skew_second, skew_first;
    ];
end