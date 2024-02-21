
L1=1;
L2=0.5;

% Angles of the first joint (q1) and the second joint (q2), set here
% physical limitations
q1 = linspace(0, pi/2, 100);
q2 = linspace(-pi/2, pi/2, 100);

% Initialize arrays for storing workspace points
workspace_x = [];
workspace_y = [];

% Loop through all possible combinations of q1 and q2
for i = 1:length(q1)
    for j = 1:length(q2)
        % Calculate the end effector position using the forward kinematics
            x_val=L1*cos(q1(i))+L2*cos(q1(i)+q2(j));
            y_val=L1*sin(q1(i))+L2*sin(q1(i)+q2(j));
            % Store the workspace points
            workspace_x = [workspace_x, x_val];
            workspace_y = [workspace_y, y_val];
    end
end

% Define the point coordinates
point_x = 0;
point_y = 0;

% Create the plot for the workspace and the point
figure;
plot(workspace_x, workspace_y, '.', 'DisplayName', 'Workspace of the End Effector');
hold on;
plot(point_x, point_y, 'rx', 'DisplayName', 'Point'); % Red point
hold off;
xlim([-3, 3]);
ylim([-3, 3]);
axis equal;
title('Workspace and Point');
xlabel('x Axis');
ylabel('y Axis');
legend;
grid on;
