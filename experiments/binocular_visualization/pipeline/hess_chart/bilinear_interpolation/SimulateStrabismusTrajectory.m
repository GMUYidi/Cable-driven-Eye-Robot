clear; clc; close all;

healthyFile = 'output_test_Yidi50222(0.01).mat';
offsetFile = 'Data for strabismus.xlsx';
sheetName = 'Q22';
patientId = sheetName;

offsetScale = 0.6;
outOfRangeMode = 'clip'; % 'clip' keeps every sample; 'nan' only keeps samples inside the Hess range.
fillMissingGridNodes = true; % The simplified Hess chart has no four corner points.
trimStartSample = 7; % Remove the initial left-up jump after the first start cluster.
verticalErrorScale = 0.25; % 0 removes L/R vertical error; 1 keeps the measured error.
outputSampleCount = 1800;
resampleMethod = 'makima'; % Smooth interpolation with less overshoot than spline.
savePlots = false;

outputDataDir = 'strabismus data';
outputBaselineDir = 'healthbaseline';
outputPrefix = ['simulated_' patientId];
outputMatFile = fullfile(outputDataDir, ['simulated_strabismus_' patientId '.mat']);
outputBaselineFile = fullfile(outputBaselineDir, ['healthbaseline_' patientId '.mat']);

if exist(outputDataDir, 'dir') ~= 7
    mkdir(outputDataDir);
end
if exist(outputBaselineDir, 'dir') ~= 7
    mkdir(outputBaselineDir);
end

rawData = load(healthyFile);
rawSampleCount = size(rawData.GazeAngleSmoothed_cell, 1);
[data, keptSamples] = trimDataStruct(rawData, trimStartSample, rawSampleCount);

originalVerticalDiff = data.GazeAngleSmoothed_cell(:, 2) - data.GazeAngleSmoothed_cell(:, 4);
data.GazeAngleSmoothed_cell = shrinkLeftRightDifference( ...
    data.GazeAngleSmoothed_cell, 2, 4, verticalErrorScale);
if isfield(data, 'GazeAngle_cell')
    data.GazeAngle_cell = shrinkLeftRightDifference(data.GazeAngle_cell, 2, 4, verticalErrorScale);
end
data.GazePointSmoothed_cell = shrinkLeftRightDifference( ...
    data.GazePointSmoothed_cell, 2, 4, verticalErrorScale);
if isfield(data, 'GazePoint_cell')
    data.GazePoint_cell = shrinkLeftRightDifference(data.GazePoint_cell, 2, 4, verticalErrorScale);
end
correctedVerticalDiff = data.GazeAngleSmoothed_cell(:, 2) - data.GazeAngleSmoothed_cell(:, 4);

Corrected294GazeAngleSmoothed_cell = data.GazeAngleSmoothed_cell;
Corrected294GazePointSmoothed_cell = data.GazePointSmoothed_cell;
[data, resampleOriginalSampleIndex] = resampleDataStruct( ...
    data, keptSamples, outputSampleCount, resampleMethod);
saveHealthBaseline(outputBaselineFile, data, rawData, ...
    Corrected294GazeAngleSmoothed_cell, keptSamples, trimStartSample, ...
    verticalErrorScale, outputSampleCount, resampleMethod, ...
    resampleOriginalSampleIndex, healthyFile, patientId);

healthyAngles = data.GazeAngleSmoothed_cell;
healthyPoints = data.GazePointSmoothed_cell;
sampleIndex = 1:size(healthyAngles, 1);

[X, Y, xGrid, yGrid, HoffsetLM, VoffsetLM, HoffsetRM, VoffsetRM] = ...
    buildOffsetGrids(offsetFile, sheetName, fillMissingGridNodes);

[leftHOffset, leftVOffset, leftWasClipped] = interpolateEyeOffsets( ...
    healthyAngles(:, 1), healthyAngles(:, 2), X, Y, xGrid, yGrid, ...
    HoffsetLM, VoffsetLM, outOfRangeMode);
[rightHOffset, rightVOffset, rightWasClipped] = interpolateEyeOffsets( ...
    healthyAngles(:, 3), healthyAngles(:, 4), X, Y, xGrid, yGrid, ...
    HoffsetRM, VoffsetRM, outOfRangeMode);

SOffsetSmoothed_cell = offsetScale * ...
    [leftHOffset, leftVOffset, rightHOffset, rightVOffset];
SGazeAngleSmoothed_cell = healthyAngles + SOffsetSmoothed_cell;

SGazePointSmoothed_cell = estimateGazePointsFromAngles( ...
    healthyAngles, healthyPoints, SGazeAngleSmoothed_cell);

fprintf('Loaded healthy trajectory: %s\n', healthyFile);
fprintf('Offset table: %s\n', offsetFile);
fprintf('Offset scale: %.3g\n', offsetScale);
fprintf('Out-of-range mode: %s\n', outOfRangeMode);
fprintf('Trimmed initial jump: kept samples %d:%d (%d samples)\n', ...
    keptSamples(1), keptSamples(end), numel(keptSamples));
fprintf('Resampled trajectory: %d -> %d samples using %s interpolation\n', ...
    numel(keptSamples), outputSampleCount, resampleMethod);
fprintf('Saved health baseline: %s\n', outputBaselineFile);
fprintf('Vertical L/R error scale: %.3g\n', verticalErrorScale);
fprintf('Vertical diff before correction: mean %.3f, std %.3f, rms %.3f\n', ...
    mean(originalVerticalDiff), std(originalVerticalDiff), sqrt(mean(originalVerticalDiff .^ 2)));
fprintf('Vertical diff after correction:  mean %.3f, std %.3f, rms %.3f\n', ...
    mean(correctedVerticalDiff), std(correctedVerticalDiff), sqrt(mean(correctedVerticalDiff .^ 2)));
fprintf('Samples clipped to Hess range: left %d/%d, right %d/%d\n', ...
    sum(leftWasClipped), numel(leftWasClipped), ...
    sum(rightWasClipped), numel(rightWasClipped));
fprintf('Samples with NaN offset: left %d/%d, right %d/%d\n', ...
    sum(any(isnan(SOffsetSmoothed_cell(:, 1:2)), 2)), size(SOffsetSmoothed_cell, 1), ...
    sum(any(isnan(SOffsetSmoothed_cell(:, 3:4)), 2)), size(SOffsetSmoothed_cell, 1));

plotEyePositionTrajectory(healthyPoints, SGazePointSmoothed_cell, outputPrefix, savePlots);
plotGazeAngleTrajectory(healthyAngles, SGazeAngleSmoothed_cell, outputPrefix, savePlots);
plotHorizontalAngle(sampleIndex, healthyAngles, SGazeAngleSmoothed_cell, outputPrefix, savePlots);
plotVerticalAngle(sampleIndex, healthyAngles, SGazeAngleSmoothed_cell, outputPrefix, savePlots);

out = data;
out.RawGazeAngleSmoothed_cell = rawData.GazeAngleSmoothed_cell;
out.RawGazePointSmoothed_cell = rawData.GazePointSmoothed_cell;
out.Corrected294GazeAngleSmoothed_cell = Corrected294GazeAngleSmoothed_cell;
out.Corrected294GazePointSmoothed_cell = Corrected294GazePointSmoothed_cell;
out.SGazeAngleSmoothed_cell = SGazeAngleSmoothed_cell;
out.SOffsetSmoothed_cell = SOffsetSmoothed_cell;
out.SGazePointSmoothed_cell = SGazePointSmoothed_cell;
out.offsetScale = offsetScale;
out.outOfRangeMode = outOfRangeMode;
out.fillMissingGridNodes = fillMissingGridNodes;
out.trimStartSample = trimStartSample;
out.keptSamples = keptSamples;
out.verticalErrorScale = verticalErrorScale;
out.outputSampleCount = outputSampleCount;
out.resampleMethod = resampleMethod;
out.resampleOriginalSampleIndex = resampleOriginalSampleIndex;
out.savePlots = savePlots;
save(outputMatFile, '-struct', 'out');
fprintf('Saved simulated trajectory: %s\n', outputMatFile);

function [trimmedData, keptSamples] = trimDataStruct(inputData, trimStartSample, sampleCount)
    keptSamples = trimStartSample:sampleCount;
    trimmedData = inputData;
    fieldNames = fieldnames(inputData);

    for i = 1:numel(fieldNames)
        fieldName = fieldNames{i};
        value = inputData.(fieldName);

        if isnumeric(value) && size(value, 1) == sampleCount
            trimmedData.(fieldName) = value(keptSamples, :);
        end
    end
end

function corrected = shrinkLeftRightDifference(values, leftColumn, rightColumn, errorScale)
    corrected = values;
    leftValue = values(:, leftColumn);
    rightValue = values(:, rightColumn);
    centerValue = (leftValue + rightValue) / 2;
    halfDifference = (leftValue - rightValue) / 2 * errorScale;
    corrected(:, leftColumn) = centerValue + halfDifference;
    corrected(:, rightColumn) = centerValue - halfDifference;
end

function [resampledData, resampleOriginalSampleIndex] = resampleDataStruct( ...
    inputData, keptSamples, outputSampleCount, method)
    inputSampleCount = numel(keptSamples);
    oldIndex = 1:inputSampleCount;
    newIndex = linspace(1, inputSampleCount, outputSampleCount);
    resampleOriginalSampleIndex = interp1(oldIndex, keptSamples, newIndex, 'linear');
    resampledData = inputData;
    fieldNames = fieldnames(inputData);

    for i = 1:numel(fieldNames)
        fieldName = fieldNames{i};
        value = inputData.(fieldName);

        if isnumeric(value) && size(value, 1) == inputSampleCount
            resampledData.(fieldName) = interp1(oldIndex, value, newIndex, method);
        end
    end
end

function saveHealthBaseline(outputBaselineFile, data, rawData, corrected294Angles, ...
    keptSamples, trimStartSample, verticalErrorScale, outputSampleCount, ...
    resampleMethod, resampleOriginalSampleIndex, healthyFile, patientId)
    baseline = struct();
    baseline.patientId = patientId;
    baseline.sourceHealthyFile = healthyFile;
    baseline.GazeAngleSmoothed_cell = data.GazeAngleSmoothed_cell;
    baseline.GazeAngle_cell = data.GazeAngle_cell;
    baseline.RawGazeAngleSmoothed_cell = rawData.GazeAngleSmoothed_cell;
    baseline.Corrected294GazeAngleSmoothed_cell = corrected294Angles;
    baseline.trimStartSample = trimStartSample;
    baseline.keptSamples = keptSamples;
    baseline.verticalErrorScale = verticalErrorScale;
    baseline.outputSampleCount = outputSampleCount;
    baseline.resampleMethod = resampleMethod;
    baseline.resampleOriginalSampleIndex = resampleOriginalSampleIndex;

    save(outputBaselineFile, '-struct', 'baseline');
end

function [X, Y, xGrid, yGrid, HoffsetLM, VoffsetLM, HoffsetRM, VoffsetRM] = ...
    buildOffsetGrids(offsetFile, sheetName, fillMissingGridNodes)
    [~, ~, raw] = xlsread(offsetFile, sheetName);
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

    if fillMissingGridNodes
        HoffsetLM = fillMissingByNeighborAverage(HoffsetLM);
        VoffsetLM = fillMissingByNeighborAverage(VoffsetLM);
        HoffsetRM = fillMissingByNeighborAverage(HoffsetRM);
        VoffsetRM = fillMissingByNeighborAverage(VoffsetRM);
    end
end

function [hOffset, vOffset, wasClipped] = interpolateEyeOffsets( ...
    targetH, targetV, X, Y, xGrid, yGrid, hOffsetGrid, vOffsetGrid, outOfRangeMode)
    queryH = targetH;
    queryV = targetV;
    wasClipped = false(size(targetH));

    switch lower(outOfRangeMode)
        case 'clip'
            clippedH = min(max(queryH, min(xGrid)), max(xGrid));
            clippedV = min(max(queryV, min(yGrid)), max(yGrid));
            wasClipped = clippedH ~= queryH | clippedV ~= queryV;
            queryH = clippedH;
            queryV = clippedV;
        case 'nan'
            outside = queryH < min(xGrid) | queryH > max(xGrid) | ...
                queryV < min(yGrid) | queryV > max(yGrid);
            queryH(outside) = NaN;
            queryV(outside) = NaN;
            wasClipped = outside;
        otherwise
            error('Unknown outOfRangeMode: %s', outOfRangeMode);
    end

    hOffset = interp2(X, Y, hOffsetGrid, queryH, queryV, 'linear', NaN);
    vOffset = interp2(X, Y, vOffsetGrid, queryH, queryV, 'linear', NaN);
end

function points = parsePointColumn(values, columnName)
    points = nan(numel(values), 2);

    for i = 1:numel(values)
        value = values{i};
        if isMissingCell(value)
            error('Missing coordinate in %s at data row %d.', columnName, i + 1);
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
        offsetGrid(yIndex, xIndex) = offsets(i);
    end
end

function filledGrid = fillMissingByNeighborAverage(inputGrid)
    filledGrid = inputGrid;

    for pass = 1:numel(inputGrid)
        missing = find(isnan(filledGrid));
        if isempty(missing)
            return;
        end

        changed = false;
        nextGrid = filledGrid;

        for idx = missing(:).'
            [row, col] = ind2sub(size(filledGrid), idx);
            neighbors = [];

            if row > 1 && isfinite(filledGrid(row - 1, col))
                neighbors(end + 1) = filledGrid(row - 1, col); %#ok<AGROW>
            end
            if row < size(filledGrid, 1) && isfinite(filledGrid(row + 1, col))
                neighbors(end + 1) = filledGrid(row + 1, col); %#ok<AGROW>
            end
            if col > 1 && isfinite(filledGrid(row, col - 1))
                neighbors(end + 1) = filledGrid(row, col - 1); %#ok<AGROW>
            end
            if col < size(filledGrid, 2) && isfinite(filledGrid(row, col + 1))
                neighbors(end + 1) = filledGrid(row, col + 1); %#ok<AGROW>
            end

            if ~isempty(neighbors)
                nextGrid(row, col) = mean(neighbors);
                changed = true;
            end
        end

        filledGrid = nextGrid;
        if ~changed
            return;
        end
    end
end

function simulatedPoints = estimateGazePointsFromAngles(healthyAngles, healthyPoints, simulatedAngles)
    simulatedPoints = nan(size(healthyPoints));
    simulatedPoints(:, 1:2) = estimateOneEyePoints( ...
        healthyAngles(:, 1:2), healthyPoints(:, 1:2), simulatedAngles(:, 1:2));
    simulatedPoints(:, 3:4) = estimateOneEyePoints( ...
        healthyAngles(:, 3:4), healthyPoints(:, 3:4), simulatedAngles(:, 3:4));
end

function simulatedEyePoints = estimateOneEyePoints(healthyEyeAngles, healthyEyePoints, simulatedEyeAngles)
    valid = all(isfinite(healthyEyeAngles), 2) & all(isfinite(healthyEyePoints), 2);
    design = [ones(sum(valid), 1), healthyEyeAngles(valid, :)];
    xCoef = design \ healthyEyePoints(valid, 1);
    yCoef = design \ healthyEyePoints(valid, 2);

    queryDesign = [ones(size(simulatedEyeAngles, 1), 1), simulatedEyeAngles];
    simulatedEyePoints = [queryDesign * xCoef, queryDesign * yCoef];
    simulatedEyePoints(any(~isfinite(simulatedEyeAngles), 2), :) = NaN;
end

function plotEyePositionTrajectory(healthyPoints, simulatedPoints, outputPrefix, savePlots)
    XL = healthyPoints(:, 1);
    YL = 1 - healthyPoints(:, 2);
    XR = healthyPoints(:, 3);
    YR = 1 - healthyPoints(:, 4);

    SXL = simulatedPoints(:, 1);
    SYL = 1 - simulatedPoints(:, 2);
    SXR = simulatedPoints(:, 3);
    SYR = 1 - simulatedPoints(:, 4);

    h = figure('Name', 'Simulated eye position trajectory', 'Color', 'w');
    plot(XL, YL, 'r--', XR, YR, 'b--', SXL, SYL, 'r-', SXR, SYR, 'b-', 'LineWidth', 1.2);
    title('Eye position trajectory');
    xlabel('Normalized X');
    ylabel('Normalized Y');
    legend({'Healthy left', 'Healthy right', 'Simulated left', 'Simulated right'}, ...
        'Location', 'bestoutside');
    xlim(visibleLimits([XL; XR; SXL; SXR], [0 1]));
    ylim(visibleLimits([YL; YR; SYL; SYR], [0 1]));
    pbaspect([8 7 1]);
    grid on;
    if savePlots
        print(h, '-djpeg', '-r300', [outputPrefix '_eye_position_trajectory.jpg']);
    end
end

function plotGazeAngleTrajectory(healthyAngles, simulatedAngles, outputPrefix, savePlots)
    h = figure('Name', 'Simulated gaze angle trajectory', 'Color', 'w');
    plot(healthyAngles(:, 1), healthyAngles(:, 2), 'r--', ...
        healthyAngles(:, 3), healthyAngles(:, 4), 'b--', ...
        simulatedAngles(:, 1), simulatedAngles(:, 2), 'r-', ...
        simulatedAngles(:, 3), simulatedAngles(:, 4), 'b-', ...
        'LineWidth', 1.2);
    title('Gaze angle trajectory');
    xlabel('Horizontal gaze angle');
    ylabel('Vertical gaze angle');
    legend({'Healthy left', 'Healthy right', 'Simulated left', 'Simulated right'}, ...
        'Location', 'bestoutside');
    axis equal;
    set(gca, 'YDir', 'reverse');
    grid on;
    if savePlots
        print(h, '-djpeg', '-r300', [outputPrefix '_gaze_angle_trajectory.jpg']);
    end
end

function plotHorizontalAngle(sampleIndex, healthyAngles, simulatedAngles, outputPrefix, savePlots)
    h = figure('Name', 'Simulated horizontal gaze angle', 'Color', 'w');
    plot(sampleIndex, healthyAngles(:, 1), 'r--', ...
        sampleIndex, healthyAngles(:, 3), 'b--', ...
        sampleIndex, simulatedAngles(:, 1), 'r-', ...
        sampleIndex, simulatedAngles(:, 3), 'b-', ...
        'LineWidth', 1.2);
    title('Horizontal gaze angle');
    xlabel('Sample index');
    ylabel('Horizontal gaze angle');
    legend({'Healthy left', 'Healthy right', 'Simulated left', 'Simulated right'}, ...
        'Location', 'bestoutside');
    grid on;
    if savePlots
        print(h, '-djpeg', '-r300', [outputPrefix '_eyeangleX.jpg']);
    end
end

function plotVerticalAngle(sampleIndex, healthyAngles, simulatedAngles, outputPrefix, savePlots)
    h = figure('Name', 'Simulated vertical gaze angle', 'Color', 'w');
    plot(sampleIndex, healthyAngles(:, 2), 'r--', ...
        sampleIndex, healthyAngles(:, 4), 'b--', ...
        sampleIndex, simulatedAngles(:, 2), 'r-', ...
        sampleIndex, simulatedAngles(:, 4), 'b-', ...
        'LineWidth', 1.2);
    title('Vertical gaze angle');
    xlabel('Sample index');
    ylabel('Vertical gaze angle');
    legend({'Healthy left', 'Healthy right', 'Simulated left', 'Simulated right'}, ...
        'Location', 'bestoutside');
    grid on;
    if savePlots
        print(h, '-djpeg', '-r300', [outputPrefix '_eyeangleY.jpg']);
    end
end

function limits = visibleLimits(values, defaultLimits)
    finiteValues = values(isfinite(values));

    if isempty(finiteValues)
        limits = defaultLimits;
        return;
    end

    dataMin = min(finiteValues);
    dataMax = max(finiteValues);
    padding = max((dataMax - dataMin) * 0.05, 0.01);
    limits = [min(defaultLimits(1), dataMin - padding), ...
        max(defaultLimits(2), dataMax + padding)];
end

function tf = isMissingCell(value)
    tf = isempty(value) || ...
        (isnumeric(value) && isscalar(value) && isnan(value)) || ...
        (isstring(value) && ismissing(value));
end
