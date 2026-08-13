function [boolRemove] = segmentOutLinesIndex(im)


    regions = regionprops(im, 'ConvexImage','ConvexArea');
    [~,i] = max([regions.ConvexArea]);
    region = regions(i);

    if isempty(region)
        boolRemove = false;
    else
        imECC = regionprops(region.ConvexImage, 'Eccentricity');


        im7 = bwmorph(im, 'branchpoints');
        %     close(gcf)
        boolRemove = true;
        if (sum(im7(:)) == 0) || (imECC.Eccentricity > 0.95)
            boolRemove = false;
        end
    end
end
