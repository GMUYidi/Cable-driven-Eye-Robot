function [x,y] = findPoints2(im)

    x = [];
    y = [];

    im5 = zeros(size(im));

    im5 = im;
%     im5 = imclose(im5,strel('disk',5));

    regions = regionprops(im5, 'Area', 'Centroid');

    [~,id] = max([regions.Area]);

    point = [regions(id).Centroid];

    if ~isempty(point)
        x = point(1);
    end
    if ~isempty(point)
        y = point(2);
    end
%     figure
%     imshow(im5)
%     hold on
%     scatter(x,y, 'filled')
%     hold off

    %% if size of y > 1, do some sort of elimination to get just 1


%     y = mean(y);
%     x = mean(x);
%     hold on
%     scatter(x,y, 'filled')
%     hold off

end
