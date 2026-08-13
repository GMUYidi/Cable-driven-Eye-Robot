%Excpects hess chart and will attempt to correct for slight rotations. More
%than 5 degrees of rotation will not be adjusted, but this can be
%adjusted in the hough function line

%INPUT:
    %im: RGB image with full page chart with left and right hess charts.

%OUTPUT
    %imRotate: the rotated image

function [imRotate] = imageRegistrationCorrectionAPP(im)


bw = imbinarize(im);
bw = sum(~bw,3);
bw2 = imfill(bw);

bwLab = bwlabel(bw2);

props = regionprops(bwLab,'Area'); %Will look for hess charts by assuming they have the highest areas
areas = [props.Area];
[~,idAS] = sort(areas,'descend');

hessBoxBW = (bwLab==idAS(1) | bwLab==idAS(2));
hessBoxBW = imopen(hessBoxBW, ones(15));

hessBW = bwmorph(hessBoxBW, 'remove');
hessBW = imdilate(hessBW, strel('square', 4));


[H,T,R] = hough(hessBW, 'Theta', 0:.001:5);

P  = houghpeaks(H,2,'threshold',ceil(0.3*max(H(:))));
x = T(P(:,2)); y = R(P(:,1));

lines = houghlines(hessBW,T,R,P,'FillGap',5,'MinLength',7);
%%
% figure, imshow(hessBoxBW), hold on
% max_len = 0;
% for k = 1:length(lines)
%    xy = [lines(k).point1; lines(k).point2];
%    plot(xy(:,1),xy(:,2),'LineWidth',5,'Color','green');
%
%    % Plot beginnings and ends of lines
%    plot(xy(1,1),xy(1,2),'x','LineWidth',5,'Color','yellow');
%    plot(xy(2,1),xy(2,2),'x','LineWidth',5,'Color','red');
%
%    % Determine the endpoints of the longest line segment
%    len = norm(lines(k).point1 - lines(k).point2);
%    if ( len > max_len)
%       max_len = len;
%       xy_long = xy;
%    end
% end
%%


avgRotation = mean([lines.theta]);

imRotate = imrotate(im, avgRotation, 'crop');

r = imRotate(:,:,1) == 0;
g = imRotate(:,:,2) == 0;
b = imRotate(:,:,3) == 0;

black = r & g & b;

black = black.*255;

imRotate = imRotate + uint8(black);

% msgbox(sprintf('rotation of %.3f degrees applied', avgRotation));
end
