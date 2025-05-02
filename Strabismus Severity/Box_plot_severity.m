% Read data
severity_values = [4.923, 2.544, 1.928, 4.027, 4.222, 5.113, 2.890, 3.976, 4.678, 7.982, 5.782, 6.723, 8.463, 7.345, 3.678, 6.986, 6.989, 5.683, 7.381, 5.612, 8.344, 3.537, 7.332, 6.664, 5.917];

% Strabismus patient data
strabismus_value = 65.109;

% Calculate maximum and minimum values
max_value = max(severity_values);
min_value = min(severity_values);

% Calculate variance
variance_value = var(severity_values);

% Plot boxplot
figure;
boxplot(severity_values, 'Labels', {'Healthy Participants'});
hold on;

% Plot strabismus patient value line
yline(strabismus_value, '--r', 'Strabismus Patient');

% Mark maximum and minimum values
plot(1, max_value, 'g*', 'MarkerSize', 10, 'DisplayName', 'Max Value');
plot(1, min_value, 'b*', 'MarkerSize', 10, 'DisplayName', 'Min Value');

% Add text annotations for maximum and minimum values
text(1.1, max_value, ['Max: ', num2str(max_value)], 'FontSize', 12, 'Color', 'g');
text(1.1, min_value, ['Min: ', num2str(min_value)], 'FontSize', 12, 'Color', 'b');

% Add text annotation for strabismus patient value
text(1.1, strabismus_value, ['Strabismus: ', num2str(strabismus_value)], 'FontSize', 12, 'Color', 'r');

% Display variance in the bottom right corner
x_limits = xlim;
y_limits = ylim;
text(x_limits(2) - 0.2, y_limits(2) +5, ['Variance: ', num2str(variance_value)], 'FontSize', 12, 'Color', 'k', 'HorizontalAlignment', 'right');

% Add title and labels
title('Severity Values of Participants');
ylabel('Severity Value');

% Display legend
legend('Healthy Participants', 'Strabismus Patient', 'Location', 'southeast');
hold off;
