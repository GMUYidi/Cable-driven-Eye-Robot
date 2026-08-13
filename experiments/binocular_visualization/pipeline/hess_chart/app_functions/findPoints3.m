function [x,y] = findPoints3(im)

    im5 = im;

    sumMatrix = @(X)(sum(X(:)));

%     meanMatrix = @(X)(mean(X(:)));


    points = im5;
    pointsFilt = nlfilter(points, [15 15], sumMatrix);

    points = points .* pointsFilt;

    points(points < max(points(:))) = 0;


    [y,x] = find(points > 0);
    if ~isempty(x)
        x = mean(x);
    end
    if ~isempty(y)
        y = mean(y);
    end
end
