clc
clear all

syms px py l2 l3 alfa

q1_pos = px - l3 * cos(alfa) + sqrt(l3^2 * cos(alfa)^2 + 2 * l3^2 * sin(alfa) * py - py^2)
q1_neg = px - l3 * cos(alfa) - sqrt(l3^2 * cos(alfa)^2 + 2 * l3^2 * sin(alfa) * py - py^2)

q2 = atan2(py/l2 - l3 * sin(alfa), px/l2 - l3 * cos(alfa))

q3 = alfa - q2