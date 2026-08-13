function [x,y] = missingPointsIt1(im)
im2 = im;

im2hsv = rgb2hsv(im2);

im3 = hsv2rgb(im2hsv.^5);

mask = segmentHueStreatchedChart(im3);

mask = imclose(mask,strel('square',8));
mask = bwareaopen(mask, 100);
mask = imclose(mask, strel('disk', 20));

mask = bwmorph(mask, 'thin', inf);

mask = imclose(mask,strel('diamond',10));

% figure
% imshow(mask)

points = mask;
points = bwmorph(points,'branchpoints');
points = imdilate(points,strel('disk', 10));

points = bwmorph(points, 'shrink', inf);


[y,x] = find(points == 1);

end
