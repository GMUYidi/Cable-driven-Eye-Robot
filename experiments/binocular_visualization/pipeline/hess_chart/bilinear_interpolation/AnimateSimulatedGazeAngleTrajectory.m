function AnimateSimulatedGazeAngleTrajectory(dataFile, animationDuration)
% Animate the simulated left/right gaze-angle trajectories.

    scriptDir = fileparts(mfilename('fullpath'));

    if nargin < 1 || isempty(dataFile)
        dataFile = fullfile( ...
            scriptDir, ...
            'strabismus data', ...
            'simulated_strabismus_Q48.mat');
    end

    if nargin < 2 || isempty(animationDuration)
        animationDuration = 15;
    end

    if animationDuration <= 0
        error('animationDuration must be positive.');
    end

    if exist(dataFile, 'file') ~= 2
        error('Data file does not exist: %s', dataFile);
    end

    data = load(dataFile, 'SGazeAngleSmoothed_cell');
    if ~isfield(data, 'SGazeAngleSmoothed_cell')
        error('SGazeAngleSmoothed_cell was not found in: %s', dataFile);
    end

    simulatedAngles = data.SGazeAngleSmoothed_cell;
    if ~isnumeric(simulatedAngles) || size(simulatedAngles, 2) < 4
        error('SGazeAngleSmoothed_cell must be an N-by-4 numeric array.');
    end

    leftAngles = simulatedAngles(:, 1:2);
    rightAngles = simulatedAngles(:, 3:4);
    sampleCount = size(simulatedAngles, 1);

    if sampleCount < 1
        error('SGazeAngleSmoothed_cell is empty.');
    end

    h = figure( ...
        'Name', 'Animated simulated gaze angle trajectory', ...
        'Color', 'w');
    ax = axes('Parent', h);
    hold(ax, 'on');

    leftLine = animatedline( ...
        ax, ...
        'Color', [0.85, 0.1, 0.1], ...
        'LineWidth', 1.6, ...
        'DisplayName', 'Simulated left');
    rightLine = animatedline( ...
        ax, ...
        'Color', [0.1, 0.25, 0.85], ...
        'LineWidth', 1.6, ...
        'DisplayName', 'Simulated right');

    title(ax, 'Gaze angle trajectory');
    xlabel(ax, 'Horizontal gaze angle');
    ylabel(ax, 'Vertical gaze angle');
    plotLegend = legend(ax, 'Location', 'none');
    plotLegend.Units = 'normalized';
    legendPosition = plotLegend.Position;
    axesPosition = ax.Position;
    legendPosition(1) = axesPosition(1) + ...
        (axesPosition(3) - legendPosition(3)) / 2;
    legendPosition(2) = axesPosition(2) + ...
        (axesPosition(4) - legendPosition(4)) / 2;
    plotLegend.Position = legendPosition;
    grid(ax, 'on');
    axis(ax, 'equal');
    set(ax, 'YDir', 'reverse');

    allHorizontal = [leftAngles(:, 1); rightAngles(:, 1)];
    allVertical = [leftAngles(:, 2); rightAngles(:, 2)];
    setAxisLimits(ax, allHorizontal, allVertical);

    addpoints(leftLine, leftAngles(1, 1), leftAngles(1, 2));
    addpoints(rightLine, rightAngles(1, 1), rightAngles(1, 2));
    drawnow;

    frameInterval = 1 / 60;
    lastSample = 1;
    animationTimer = tic;

    while isvalid(h)
        elapsed = toc(animationTimer);
        progress = min(elapsed / animationDuration, 1);
        nextSample = 1 + floor(progress * (sampleCount - 1));

        if nextSample > lastSample
            newSamples = (lastSample + 1):nextSample;
            addpoints( ...
                leftLine, ...
                leftAngles(newSamples, 1), ...
                leftAngles(newSamples, 2));
            addpoints( ...
                rightLine, ...
                rightAngles(newSamples, 1), ...
                rightAngles(newSamples, 2));
            lastSample = nextSample;
        end

        drawnow limitrate;

        if progress >= 1
            break;
        end

        pause(frameInterval);
    end

    if isvalid(h) && lastSample < sampleCount
        newSamples = (lastSample + 1):sampleCount;
        addpoints(leftLine, leftAngles(newSamples, 1), leftAngles(newSamples, 2));
        addpoints(rightLine, rightAngles(newSamples, 1), rightAngles(newSamples, 2));
        drawnow;
    end
end

function setAxisLimits(ax, horizontalValues, verticalValues)
    horizontalValues = horizontalValues(isfinite(horizontalValues));
    verticalValues = verticalValues(isfinite(verticalValues));

    if isempty(horizontalValues) || isempty(verticalValues)
        xlim(ax, [-1, 1]);
        ylim(ax, [-1, 1]);
        return;
    end

    xLimits = paddedLimits(horizontalValues);
    yLimits = paddedLimits(verticalValues);
    xlim(ax, xLimits);
    ylim(ax, yLimits);
end

function limits = paddedLimits(values)
    valueMin = min(values);
    valueMax = max(values);
    valueRange = valueMax - valueMin;

    if valueRange == 0
        padding = max(abs(valueMin) * 0.05, 1);
    else
        padding = valueRange * 0.05;
    end

    limits = [valueMin - padding, valueMax + padding];
end
