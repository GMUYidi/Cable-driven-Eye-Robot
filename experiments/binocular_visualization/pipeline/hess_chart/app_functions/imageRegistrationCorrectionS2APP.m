%Excpects hess chart and will attempt to correct for slight rotations. More
%than 5 degrees of rotation will not be adjusted atm, but this can be
%sdjusted in the hough function line

%INPUT:
    %im: RGB image, single hess chart (either left or right eye) NOT the
    %full chart.
%OUTPUT:
    %imRot: rotated image

function [imRot] = imageRegistrationCorrectionS2APP(im)


bw = imbinarize(im);
bw = sum(~bw,3);
bw2 = imfill(bw);

bwLab = bwlabel(bw2);

props = regionprops(bwLab,'Area'); %Will look for hess charts by assuming they have the highest areas
areas = [props.Area];
[~,idAS] = sort(areas,'descend');

hessBoxBW = (bwLab==idAS(1));
hessBoxBW = imopen(hessBoxBW, ones(15));

hessBW = bwmorph(hessBoxBW, 'remove');
hessBW = imdilate(hessBW, strel('square', 4));


[H,T,R] = hough(hessBW, 'Theta', 0:.01:5);


P  = houghpeaks(H,5,'threshold',ceil(0.5*max(H(:))));
x = T(P(:,2)); y = R(P(:,1));

lines = houghlines(hessBW,T,R,P,'FillGap',5,'MinLength',7);

avgRotation = mean([lines.theta]);


bwRot = imrotate(hessBW, avgRotation, 'crop');
propsRot = regionprops(bwRot,'Area', 'BoundingBox');

bb = round(propsRot(1).BoundingBox);

xRange = bb(1):bb(3);
yRange = bb(2):bb(4);

imRot = imrotate(im, avgRotation, 'crop');
imRot = imRot(yRange,xRange,:);

r = imRot(:,:,1) == 0;
g = imRot(:,:,2) == 0;
b = imRot(:,:,3) == 0;

black = r & g & b;

black = black.*255;

imRot = imRot + uint8(black);

% msgbox(sprintf('rotation of %.3f degrees applied', avgRotation));
end
