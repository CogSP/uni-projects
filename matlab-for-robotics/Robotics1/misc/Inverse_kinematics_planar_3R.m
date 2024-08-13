%se la somma phi = q1 + q2 + q3 non è specificata allora ci sono infinite
%soluzioni per questo problema 

syms q1 q2

p_x = 1 + 0.5*cos(2*pi*0.25); 
p_y = 1 + 0.5*sin(2*pi*0.25); 
L1 = 1; 
L2 = 1; 
L3 = 1; 
phi = 2*pi*0.25; 

p_wx = p_x - L3*cos(phi);
p_wy = p_y - L3*sin(phi);

cos_q2 = (p_wx^2 + p_wy^2 - L1^2 - L2^2) / (2*L1*L2);
sin_q2_plus = sqrt(1 - cos_q2^2);
sin_q2_neg = -sqrt(1 - cos_q2^2);

q2_plus = atan2(sin_q2_plus,cos_q2);
q2_neg = atan2(sin_q2_neg,cos_q2);

sin_q1_plus = (p_wy*(L1 + L2*cos_q2) - L2*sin_q2_plus*p_wx) / (p_wx^2 + p_wy^2);
sin_q1_neg = (p_wy*(L1 + L2*cos_q2) - L2*sin_q2_neg*p_wx) / (p_wx^2 + p_wy^2);

cos_q1_plus = (p_wx*(L1 + L2*cos_q2) + L2*sin_q2_plus*p_wy) / (p_wx^2 + p_wy^2);
cos_q1_neg = (p_wx*(L1 + L2*cos_q2) + L2*sin_q2_neg*p_wy) / (p_wx^2 + p_wy^2);

q1_plus = atan2(sin_q1_plus,cos_q1_plus);
q1_neg = atan2(sin_q1_neg,cos_q1_neg);

q3_plus = phi - q1_plus - q2_plus;
q3_neg = phi - q1_neg - q2_neg;

disp("First solution:");
q_first = [q1_plus; q2_plus; q3_plus];
disp(q_first);

disp("Second solution:");
q_second = [q1_neg; q2_neg; q3_neg];
disp(q_second);
