theta = pi/2;
%vector r = [rx ry rz]
r = [-0.3536
   -0.8660
   -0.3536];

r11 = r(1)^2*(1 - cos(theta)) + cos(theta);
r22 = r(2)^2*(1 - cos(theta)) + cos(theta);
r33 = r(3)^2*(1 - cos(theta)) + cos(theta);

r12 = r(1)*r(2)*(1 - cos(theta)) - r(3)*sin(theta);
r21 = r(1)*r(2)*(1 - cos(theta)) + r(3)*sin(theta);
r31 = r(1)*r(3)*(1 - cos(theta)) - r(2)*sin(theta);

r32 = r(2)*r(3)*(1 - cos(theta)) + r(1)*sin(theta);
r13 = r(1)*r(3)*(1 - cos(theta)) + r(2)*sin(theta);
r23 = r(2)*r(3)*(1 - cos(theta)) - r(1)*sin(theta);

R = [r11 r12 r13;
    r21 r22 r23;
    r31 r32 r33;];

disp("R = ");
disp(R);