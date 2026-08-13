%attempts to segment the pencil marks in the hess chart. Initial
%segmentation is done by quantizing the chart colors, and then using an
%assosiated segmentation function to select pencil markings,
%Binary image manipulations are then done to try and remove text and other
%misc features.
%Ideally, a segmentation method should be made to remove the text from the
%charts as well.

%INPUTS:
    %im: input image, should be RGB, no alpha channel
    %nQ: integer the specifies how many colors should be in the quantized
    %version of the image
    %segmentFunction: Function handle that gets a mask of the quantized
    %image. should be compatible with whatever your nQ is.
%OUTPUTS:
    %maskOut: segmented and processed mask that ideally has the pencil
    %markings from the hess chart

function [maskOut] = getChartLines(im, nQ, segmentFunction)

%       convert images to double. Probably not strictly necessary but
%       didn't want to find the appropriate function
        im2 = rgb2hsv(im);
        im3 = hsv2rgb(im2);

        %quantize the images with
        [im4, cm] = rgb2ind(1 - im3,nQ);
        im5 = modefilt(im4,[1 1]);

        im6 = ind2rgb(im5,cm);
        mask0 = segmentFunction(im6);

        mask = imclose(mask0, strel('disk',7));
        mask = bwmorph(mask, 'clean');
        mask = bwmorph(mask,'bridge',inf);
        mask = modefilt(mask,[3 3]);
        mask = imclose(mask, strel('disk',7));


        mask1 = zeros(size(mask));

        %Will remove anything that is not long enough in a particular
        %angle
        for d = 1:180
            maskd = imopen(mask,strel('line',50,d));
            mask1 = mask1 + maskd;
        end

        mask1 = bwmorph(mask1,'thin',inf);
        mask1 = bwmorph(mask1, 'clean');


        maskFill = imfill(imdilate(mask1,strel('disk', 15)),'holes');
        maskFill = imerode(maskFill,strel('disk',15));
        maskFill = imopen(maskFill,strel('disk',11));

        maskEdge = bwconvhull(maskFill);
        maskEdge = edge(maskEdge);

        mask2 = mask1 | maskEdge;
        mask2 = imerode(~mask2,strel('disk',3));
        mask2 = ~bwareaopen(mask2,5000);
        mask2 = bwmorph(mask2,'shrink',inf);

       mask3 = bwlabel(imerode(~mask2,strel('disk',3)));
       mask4 = zeros(size(mask3));
       for iii = 1:max(mask3(:))
           mask3temp = mask3 == iii;
           mask4 = mask4 + imclose(mask3temp,strel('disk',10));

       end

       mask5 = bwmorph(~mask4,'thin', inf);
       maskOut = mask5;
end
