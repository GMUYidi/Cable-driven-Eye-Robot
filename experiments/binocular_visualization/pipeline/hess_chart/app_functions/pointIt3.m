function nPoints = pointIt3(im, points, nHood)

    for i = 1:2

        [subImages, Spacers] = getSubImagesFromCenterAPP(im{i}, points{i}, nHood);


        xf = [];
        yf = [];

        for sss = 1:length(subImages)



            [x,y] = findPoints3(subImages{sss});


            xf = [xf (x'+Spacers(sss,2)-1)];
            yf = [yf (y'+Spacers(sss,1)-1)];


        end
        nPoints{i} = [xf',yf'];
    end


end
