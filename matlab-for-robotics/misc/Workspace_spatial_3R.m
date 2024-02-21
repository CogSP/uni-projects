N=0.3;

% Angles of the first joint (q1), the second joint (q2), and the third joint (q3)
q1 = linspace(0, 1.0, 30);
q2 = linspace(-pi, pi, 30);
q3 = linspace(0, 1.0, 30);

% Initialize arrays for storing workspace points
workspace_x = [];
workspace_y = [];
workspace_z = [];

% Loop through all possible combinations of q1, q2, and q3
for i = 1:length(q1)
    for j = 1:length(q2)
        for k = 1:length(q3)
            % Calculate the end effector position using the forward kinematics
            x_val = N * cos(q2(j)) - q3(k) * sin(q2(j));
            y_val = N * sin(q2(j)) + q3(k) * cos(q2(j));
            z_val = q1(i);

            % Store the workspace points
            workspace_x = [workspace_x, x_val];
            workspace_y = [workspace_y, y_val];
            workspace_z = [workspace_z, z_val];
        end
    end
end

% Define the point coordinates
point_x = 0;
point_y = 0;
point_z = 0; % Assuming a specific value for q3

% Create the 3D plot for the workspace, the point, and real eigenvectors
figure;
plot3(workspace_x, workspace_y, workspace_z, '.', 'DisplayName', 'Workspace of the End Effector');
hold on;
plot3(point_x, point_y, point_z, 'rx', 'DisplayName', 'Point'); % Red point
hold off;
xlabel('x Axis');
ylabel('y Axis');
zlabel('z Axis (q3)');
title('3D Workspace, Point');
legend;
grid on;

