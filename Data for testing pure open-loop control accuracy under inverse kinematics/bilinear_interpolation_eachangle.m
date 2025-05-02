%data = dlmread('data.txt');
data = dlmread('data_test_open_loop.txt');

yaw_actual = data(:, 1);
yaw_target = data(:, 4);
pitch_actual = data(:, 2);
pitch_target = data(:, 5);
row_actual = data(:, 3);
row_target = data(:, 6);

yaw_error = yaw_actual - yaw_target;
pitch_error = pitch_actual - pitch_target;
row_error = row_actual - row_target;

x = -10:0.5:10; 
y = -10:0.5:10; 


[X, Y] = meshgrid(x, y);


Z_yaw = griddata(pitch_actual, yaw_actual, yaw_error, X, Y, 'linear');
Z_pitch = griddata(pitch_actual, yaw_actual, pitch_error, X, Y, 'linear');
Z_row = griddata(pitch_actual, yaw_actual, row_error, X, Y, 'linear');

min_error = min([yaw_error; pitch_error; row_error]);
max_error = max([yaw_error; pitch_error; row_error]);


% 设置字体大小
fontSize = 14; % 字体大小
titleSize = 16; % 标题字体大小

figure;
contourf(X, Y, Z_yaw, 20, 'LineColor', 'none');
caxis([min_error max_error]); 
colorbar;
xlabel('Yaw angle (degrees)', 'FontSize', fontSize);
ylabel('Pitch angle (degrees)', 'FontSize', fontSize);
title('Yaw error heatmap', 'FontSize', titleSize);
set(gca, 'FontSize', fontSize); % 设置坐标轴字体大小

figure;
contourf(X, Y, Z_pitch, 20, 'LineColor', 'none'); 
caxis([min_error max_error]); 
colorbar; 
xlabel('Yaw angle (degrees)', 'FontSize', fontSize);
ylabel('Pitch angle (degrees)', 'FontSize', fontSize);
title('Pitch error heatmap', 'FontSize', titleSize);
set(gca, 'FontSize', fontSize); % 设置坐标轴字体大小

figure;
contourf(X, Y, Z_row, 20, 'LineColor', 'none'); 
caxis([min_error max_error]); 
colorbar; 
xlabel('Yaw angle (degrees)', 'FontSize', fontSize);
ylabel('Pitch angle (degrees)', 'FontSize', fontSize);
title('Roll error heatmap', 'FontSize', titleSize);
set(gca, 'FontSize', fontSize); % 设置坐标轴字体大小
