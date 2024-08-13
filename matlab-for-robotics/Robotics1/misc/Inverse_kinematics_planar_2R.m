p_x = 3/sqrt(2);
p_y = -1/sqrt(2); 

l1 = 2; 
l2 = 1; 

c2 = (p_x^2 + p_y^2 - (l1^2 + l2^2)) / (2*l1*l2);

s2_pos = sqrt(1 - c2^2);
s2_neg = -sqrt(1 - c2^2);

q2_pos = atan2(s2_pos,c2);
q2_neg = atan2(s2_neg,c2);

q1_pos = atan2(p_y,p_x) - atan2(l2*s2_pos, l1 + l2*c2);
q1_neg = atan2(p_y,p_x) - atan2(l2*s2_neg, l1 + l2*c2);

q_first = [q1_pos; q2_pos];
q_second = [q1_neg; q2_neg];

disp("First solution:");
disp(q_first);
disp("Second solution:");
disp(q_second);
