syms alpha beta gamma

rot_x = [1, 0, 0;
         0, cos(alpha), -sin(alpha);
         0, sin(alpha), cos(alpha);]

rot_y = [cos(beta), 0, sin(beta);
         0, 1, 0;
         -sin(beta), 0, cos(beta);]

rot_z = [cos(gamma), -sin(gamma), 0;
         sin(gamma), cos(gamma), 0;
         0, 0, 1;]

