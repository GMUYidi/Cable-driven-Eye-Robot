%Holdover function from a previous iteration, can probably be removed to make
%the child function 'getPointsIndexIt1' the function called in the application


function points = getHessPointsIt1(imAll,nQ,segmentFunction)
    for i = 1:2
        %% Get Eyetrack Marks
        %do image manipulations to get the gray graphite markings that show the eye
        %locations. Needs to be tested for robustness

        cIM = imAll{i};
        [x,y] = getPointsIndexIt1(cIM,nQ,segmentFunction);


        points{i} = [x,y];
    end
end
