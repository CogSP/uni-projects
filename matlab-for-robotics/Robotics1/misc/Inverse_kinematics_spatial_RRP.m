p_x = 0; 
p_y = 0; 
p_z = 0; 

d_1 = 0;  %distance to base and first joint 

q3 = sqrt(p_x^2 + p_y^2 + (p_z - d_1)^2);

s2 = (p_z - d1) / q3; 
c2_pos = sqrt(p_x^2 + p_y^2) / q3;
c2_neg = -sqrt(p_x^2 + p_y^2) / q3;
q2_pos = atan2(s2,c2_pos);
q2_neg = atan2(s2,c2_neg);

q1_pos = atan2(p_y / c2_pos, p_x/c2_pos);
q1_neg = atan2(p_y / c2_neg, p_x/c2_neg);

q_pos = [q1_pos, q2_pos, q3];
q_neg = [q1_neg, q2_neg, q3];

disp("First solution:");
disp(q_pos);
disp("Second solution:");
disp(q_neg);

