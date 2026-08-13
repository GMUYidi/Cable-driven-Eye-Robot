%Takes an input image and set of coordinates on that image, and returns a
%cell array of sub images centered on each of those coordinates.

%INPUT:
    %im: image to be processed. can be any 3 dimensional format
    %points: n,2 array with x and y coordinates
    %nHood: integer that determines the size of the sub image arround a
    %coordinate. This sub image will be square.
%OUTPUT:
    %sI: cell array of sub images, each centered on a coordinate.
    %Spacers: 1,2 array that keeps track of where the sub image is in the
    %original image; [row, column]


function [sI,Spacers] = getSubImagesFromCenterAPP(im,points, nHood)

    imwidth = nHood;
    imheight = nHood;


    for ppp = 1:size(points,1)

        xRange = floor(points(ppp,1) - imwidth/2) : floor(points(ppp,1) + imwidth/2);
        yRange = floor(points(ppp,2) - imheight/2) : floor(points(ppp,2) + imheight/2);

        if xRange(1) <= 0
            xRange = xRange-(xRange(1))+1;
        end
        if yRange(1) <= 0
            yRange = yRange-(yRange(1))+1;
        end
        if xRange(end) > size(im,2)
            xRange = xRange-(xRange(end)-size(im,2));
        end
        if yRange(end) > size(im,1)
            yRange = yRange-(yRange(end)-size(im,1));
        end

        sI{ppp} = im(yRange,xRange,:);

        Spacers(ppp,:) = [yRange(1) xRange(1)];
    end

end
