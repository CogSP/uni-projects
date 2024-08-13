clc
clear all

syms px py l alfa

q2_pos = + sqrt((px - l * cos(alfa))^2 + (py - l * sin(alfa))^2) 
q2_neg = - sqrt((px - l * cos(alfa))^2 + (py - l * sin(alfa))^2) 

q1 = atan2(py - l * sin(alfa), px - l * cos(alfa))

q3 = alfa - q1