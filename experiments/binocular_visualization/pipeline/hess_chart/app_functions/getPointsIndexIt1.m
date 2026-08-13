%Performes a lenient segmentation of the hess chart, and looks for points
%that may be intersections. This is done deterministically. It will likely include
%many extra points that are not interesection points, but the goal is that
%intersection points will be included in the set of detected points, and
%future iterations will select out the unnecessary points.

%INPUTS:
    %im: input image, should be RGB, no alpha channel
    %nQ: integer the specifies how many colors should be in the quantized
    %version of the image
    %segmentFunction: Function handle that gets a mask of the quantized
    %image. should be compatible with whatever your nQ is.
%OUTPUTS:
    %x,y: x and y position of the detected points
function [x,y] = getPointsIndexIt1(im,nQ,segmentFunction)
    im2 = rgb2hsv(im);
    im = imsharpen(hsv2rgb(im2));

    [im4, cm] = rgb2ind(1 - im,nQ);

    im5 = modefilt(im4,[1 1]);

    im6 = ind2rgb(im5,cm);

    mask = segmentFunction(im6);

    mask2 = bwmorph(mask,'clean');
    mask2 = bwmorph(mask2,'majority');
    mask2 = imclose(mask2,strel('disk',15));

    mask3 = bwmorph(mask2,'thin',inf);
    mask3 = bwareaopen(mask3,50);

    mask4 = bwmorph(mask3,'branchpoints');

    [y,x] = find(mask4 == 1);

end
