%% Compute pseudoinverse of Jacobian Matrix

function Jpse = pseudoinverse(J,m,n)
% J is Jacobian Matrix
% m is the dimension of space, for example if robot is planare we have m=2
% n is dimension of joint variables

rho = min(m,n)
if(rank(J) <= rho )
    temp = J * transpose(J)
    Jpse = transpose(J) * inv(temp)
else
    temp = transpose(J) * J;
    Jpse = inv(temp) * transpose(J);
end