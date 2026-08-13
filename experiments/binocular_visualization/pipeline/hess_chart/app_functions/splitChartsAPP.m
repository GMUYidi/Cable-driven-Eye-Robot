function [imAll, bbs] = splitChartsAPP(im)
            %% Find bounding Boxes of both charts
            bw = imbinarize(im);
            bw = sum(~bw,3);
            bw2 = imfill(bw);

            bwLab = bwlabel(bw2);

            props = regionprops(bwLab,'Area'); %Will look for hess charts by assuming they have the highest areas
            areas = [props.Area];
            [~,idAS] = sort(areas,'descend');

            hessBoxBW = (bwLab==idAS(1) | bwLab==idAS(2));
            hessBoxBW = imopen(hessBoxBW, ones(15));


            props2 = regionprops(hessBoxBW, 'BoundingBox');
            bbs = [props2(1).BoundingBox; props2(2).BoundingBox];

            %% Split Image into Right and Left Eye

            imL = im(floor(bbs(1,2)):floor(bbs(1,2)+bbs(1,4)),floor(bbs(1,1)):floor(bbs(1,1)+bbs(1,3)),:);
            imR = im(floor(bbs(2,2)):floor(bbs(2,2)+bbs(2,4)),floor(bbs(2,1)):floor(bbs(2,1)+bbs(2,3)),:);

            imAll{1} = imL;%imresize(imL,[1000 1000]);
            imAll{2} = imR;%imresize(imR,[1000 1000]);
        end
