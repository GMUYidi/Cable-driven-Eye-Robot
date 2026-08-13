function [nPoints] = RemoveLineSubImagesIt3QC(im, points, nHood)
for i = 1:2
%         im2 = rgb2hsv(im{i});
%         im3 = imsharpen(hsv2rgb(im2));
%
%
%         [im4, cm] = rgb2ind(1 - im3,nQ);
%
%         im5 = modefilt(im4,[1 1]);
%
%         im6 = ind2rgb(im5,cm);
%
%         mask0 = segmentFunction(im6);
%         mask = bwmorph(mask0,'bridge');
%         mask = modefilt(mask,[3 3]);
%         mask = imclose(mask,strel('disk',10));
%         mask = bwmorph(mask,'skel',inf);
%
%         mask2 = bwmorph(~bwareaopen(imerode(~mask,strel('disk',3)),3000),'shrink',inf);
%
%
%         mask2 = bwmorph(mask2, 'clean');
%         mask2 = imclose(mask2,strel('disk',15));
%         mask2 = bwmorph(mask2, 'thin', inf);
% %         mask2 = imclose(mask2, strel('disk', 3));
% %         figure
% %         imshow(mask2)

        [subImages, Spacers] = getSubImagesFromCenterAPP(im{i}, points{i}, nHood);
        %         [subImages, Spacers] = getSubImagesFromCenter(hessInfo.imAll{i}, [375 548], nHood);

        boolVals = ones(size(points{i},1),1);
%         hessInfo.ChartName
%         f = uifigure;
%         wb = uiprogressdlg(f, 'Value', 0,'Message','Start');
        for sss = 1:length(subImages)

        boolVals(sss) = segmentOutLinesIndex(subImages{sss});
%         cText = sprintf('point %d/%d',sss,length(subImages));
%         wb.Message = cText;
%         wb.Value = sss./length(subImages);

        end
        nPoints{i} = points{i}(logical(boolVals),:);
%         close(f)
    end


end
