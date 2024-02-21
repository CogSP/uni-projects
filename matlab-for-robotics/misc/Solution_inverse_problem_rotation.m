% Inverse problem
% Calculate the difference R - R^T
R=[  -0.4356,   -0.6597,    0.6124;
    0.4330,   -0.7500,   -0.5000;
    0.7891,    0.0474,    0.6124];

result = R - transpose(R);

sin_theta = (1 / 2) * sqrt(result(1,2)^2 + result(1,3)^2 + result(2,3)^2);
theta = atan2(sqrt(result(1,2)^2 + result(1,3)^2 + result(2,3)^2), R(1,1) + R(2,2) + R(3,3) - 1)

r = 0;
r1 = 0;
r2 = 0;

if theta == 0
    disp('No solution');
    
elseif theta == pi || theta==-pi
    rx_ry = R(1,2)/2;
    rx_rz = R(1,3)/2;
    ry_rz = R(2,3)/2;
    disp('r_23 /2')
    disp(ry_rz)
    
    r1 = [sqrt((R(1,1) + 1)/2), sqrt((R(2,2) + 1)/2), sqrt((R(3,3) + 1)/2)];
    r2 = [-sqrt((R(1,1) + 1)/2), -sqrt((R(2,2) + 1)/2), -sqrt((R(3,3) + 1)/2)];
    % since ry_rz has minus sign, ry and rz has opposite signs
    % so r1 and r2 has + and -
    disp('ry_rz');
    disp(r1(1,2)*r1(1,3));
    if ry_rz < 0 
        if(r1(1,2) > 0)
            r1(1,3) = -abs(r1(1,3));
            r2(1,2) = -abs(r2(1,2));
            r2(1,3) = abs(r2(1,3));
        else
            r1(1,3) = abs(r1(1,3));
            r2(1,2) = abs(r2(1,2));
            r2(1,3) = -abs(r2(1,3));
        end
    end
    if rx_ry < 0
        if(r1(1,1) > 0)
            r1(1,2) = -abs(r1(1,2));
            r2(1,1) = -abs(r2(1,1));
            r2(1,2) = abs(r2(1,2));
        else
            r1(1,2) = abs(r1(1,2));
            r2(1,1) = abs(r2(1,1));
            r2(1,2) = -abs(r2(1,2));
        end
    end
    if rx_rz < 0
        if(r1(1,3) > 0)
            r1(1,1) = -abs(r1(1,1));
            r2(1,3) = -abs(r2(1,3));
            r2(1,1) = abs(r2(1,1));
        else
            r1(1,1) = abs(r1(1,1));
            r2(1,3) = abs(r2(1,3));
            r2(1,1) = -abs(r2(1,1));
        end
    end

    if ry_rz > 0 
        if(r1(1,2) > 0)
            r1(1,3) = abs(r1(1,3));
            r2(1,2) = -abs(r2(1,2));
            r2(1,3) = -abs(r2(1,3));
        else
            r1(1,3) = -abs(r1(1,3));
            r2(1,2) = abs(r2(1,2));
            r2(1,3) = abs(r2(1,3));
        end
    end
    if rx_ry > 0
        if(r1(1,1) > 0)
            r1(1,2) = abs(r1(1,2));
            r2(1,1) = -abs(r2(1,1));
            r2(1,2) = -abs(r2(1,2));
        else
            r1(1,2) = -abs(r1(1,2));
            r2(1,1) = abs(r2(1,1));
            r2(1,2) = abs(r2(1,2));
        end
    end
    if rx_rz > 0
        if(r1(1,3) > 0)
            r1(1,1) = abs(r1(1,1));
            r2(1,3) = -abs(r2(1,3));
            r2(1,1) = -abs(r2(1,1));
        else
            r1(1,1) = -abs(r1(1,1));
            r2(1,3) = abs(r2(1,3));
            r2(1,1) = abs(r2(1,1));
        end
    end
disp('r1')
disp(r1);
disp('r2')
disp(r2)
else
    % regular case, there will be two opposite solution
    %(r, theta)
    %(-r, -theta)
    r1 = (1 / (2*sin(theta))) * [result(3,2); result(1,3); result(2,1)]
    r2 = -r1
end