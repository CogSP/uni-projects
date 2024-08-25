function cross_product = spatial_cross_product(first_vector, second_vector)
    omega = first_vector(1:3);
    v = first_vector(4:6);

    v_skew = skew(v);
    omega_skew = skew(omega);

    temp = [
        omega_skew, zeros(3, 3);
        v_skew, omega_skew;
    ];

    cross_product = temp * second_vector;
end
