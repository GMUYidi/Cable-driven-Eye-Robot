

function points = getMissingPoints(imAll)
    for i = 1:2
        %% Get Eyetrack Marks
        %do image manipulations to get the gray graphite markings that show the eye
        %locations. Needs to be tested for robustness

        cIM = imAll{i};
        [x,y] = missingPointsIt1(cIM);


        points{i} = [x,y];
    end
end
