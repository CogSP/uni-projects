function J_dot = get_J_dot(J, q, q_dot)
    J_dot = diff(J, q(1)) * q_dot(1);
    for i = 2:numel(q)
        J_dot = J_dot + diff(J, q(i)) * q_dot(i);
    end
end