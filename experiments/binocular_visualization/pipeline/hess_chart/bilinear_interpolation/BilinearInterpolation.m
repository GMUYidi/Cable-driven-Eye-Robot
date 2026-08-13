clear; clc; close all;

dataFile = 'Data for strabismus.xlsx';
sheetName = 'Q3';

[~, ~, raw] = xlsread(dataFile, sheetName);
raw = raw(2:end, 1:4);
raw = raw(~all(cellfun(@isMissingCell, raw), 2), :);

leftTarget = parsePointColumn(raw(:, 1), 'left eye(target)');
leftActual = parsePointColumn(raw(:, 2), 'left eye(actual)');
rightTarget = parsePointColumn(raw(:, 3), 'right eye(target)');
rightActual = parsePointColumn(raw(:, 4), 'right eye(actual)');

leftOffset = leftActual - leftTarget;
rightOffset = rightActual - rightTarget;

xGrid = unique([leftTarget(:, 1); rightTarget(:, 1)]).';
yGrid = unique([leftTarget(:, 2); rightTarget(:, 2)]).';
[X, Y] = meshgrid(xGrid, yGrid);

HoffsetLM = makeOffsetGrid(leftTarget, leftOffset(:, 1), xGrid, yGrid);
VoffsetLM = makeOffsetGrid(leftTarget, leftOffset(:, 2), xGrid, yGrid);
HoffsetRM = makeOffsetGrid(rightTarget, rightOffset(:, 1), xGrid, yGrid);
VoffsetRM = makeOffsetGrid(rightTarget, rightOffset(:, 2), xGrid, yGrid);

[Xq, Yq] = meshgrid(min(xGrid):0.25:max(xGrid), min(yGrid):0.25:max(yGrid));

HLQ = interp2(X, Y, HoffsetLM, Xq, Yq, 'linear', NaN);
VLQ = interp2(X, Y, VoffsetLM, Xq, Yq, 'linear', NaN);
HRQ = interp2(X, Y, HoffsetRM, Xq, Yq, 'linear', NaN);
VRQ = interp2(X, Y, VoffsetRM, Xq, Yq, 'linear', NaN);

% Interpolated actual gaze angle maps. Query points with NaN offset are outside
% the bilinear interpolation support of the measured Hess points.
leftActualHq = Xq + HLQ;
leftActualVq = Yq + VLQ;
rightActualHq = Xq + HRQ;
rightActualVq = Yq + VRQ;

queryLeftActualGaze = @(targetH, targetV) queryActualGaze( ...
    targetH, targetV, X, Y, HoffsetLM, VoffsetLM);
queryRightActualGaze = @(targetH, targetV) queryActualGaze( ...
    targetH, targetV, X, Y, HoffsetRM, VoffsetRM);

% Example query for one arbitrary target gaze angle after running the script:
% [leftH, leftV] = queryLeftActualGaze(10, 5);
% [rightH, rightV] = queryRightActualGaze(10, 5);

figure('Name', 'Strabismus offset from simplified Hess data', 'Color', 'w');
plotOffsetSurface(1, Xq, Yq, HLQ, leftTarget, leftOffset(:, 1), 'Horizontal offset Left eye');
plotOffsetSurface(2, Xq, Yq, VLQ, leftTarget, leftOffset(:, 2), 'Vertical offset Left eye');
plotOffsetSurface(3, Xq, Yq, HRQ, rightTarget, rightOffset(:, 1), 'Horizontal offset Right eye');
plotOffsetSurface(4, Xq, Yq, VRQ, rightTarget, rightOffset(:, 2), 'Vertical offset Right eye');
colormap(jet);

function points = parsePointColumn(values, columnName)
    points = nan(numel(values), 2);

    for i = 1:numel(values)
        value = values{i};
        if isMissingCell(value)
            error('Missing coordinate in %s at data row %d.', columnName, i + 1);
        end

        if isnumeric(value) && numel(value) == 2
            points(i, :) = value(:).';
            continue;
        end

        textValue = char(string(value));
        tokens = regexp(textValue, '[-+]?\d*\.?\d+', 'match');

        if numel(tokens) ~= 2
            error('Could not parse %s at data row %d: %s', columnName, i + 1, textValue);
        end

        points(i, :) = str2double(tokens);
    end

    % Hess chart Excel pitch values use the opposite vertical axis convention.
    points(:, 2) = -points(:, 2);
end

function offsetGrid = makeOffsetGrid(targetPoints, offsets, xGrid, yGrid)
    offsetGrid = nan(numel(yGrid), numel(xGrid));

    for i = 1:size(targetPoints, 1)
        xIndex = find(abs(xGrid - targetPoints(i, 1)) < 1e-9, 1);
        yIndex = find(abs(yGrid - targetPoints(i, 2)) < 1e-9, 1);

        if isempty(xIndex) || isempty(yIndex)
            error('Target point (%g, %g) is not on the interpolation grid.', ...
                targetPoints(i, 1), targetPoints(i, 2));
        end

        offsetGrid(yIndex, xIndex) = offsets(i);
    end
end

function plotOffsetSurface(plotIndex, Xq, Yq, offsetQ, targetPoints, measuredOffsets, plotTitle)
    subplot(2, 2, plotIndex);
    surf(Xq, Yq, offsetQ, 'EdgeColor', 'none');
    hold on;
    scatter3(targetPoints(:, 1), targetPoints(:, 2), measuredOffsets, 30, ...
        'k', 'filled', 'MarkerEdgeColor', 'w');
    hold off;
    title(plotTitle);
    xlabel('Target horizontal gaze angle');
    ylabel('Target vertical gaze angle');
    zlabel('Offset');
    colorbar;
    grid on;
    axis tight;
end

function [actualH, actualV] = queryActualGaze(targetH, targetV, X, Y, HoffsetGrid, VoffsetGrid)
    hOffset = interp2(X, Y, HoffsetGrid, targetH, targetV, 'linear', NaN);
    vOffset = interp2(X, Y, VoffsetGrid, targetH, targetV, 'linear', NaN);
    actualH = targetH + hOffset;
    actualV = targetV + vOffset;
end

function tf = isMissingCell(value)
    tf = isempty(value) || ...
        (isnumeric(value) && isscalar(value) && isnan(value)) || ...
        (isstring(value) && ismissing(value));
end
