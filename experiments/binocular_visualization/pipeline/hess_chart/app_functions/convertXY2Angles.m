%Points should be on a range of -1:1 in both directions
%Equations derived by reverse engineering equations done here:
%https://math.stackexchange.com/questions/2660373/hess-coordinate-transformation-sphere-projection

%INPUT:
        %points: n,2 array where column 1 is the x position, and column 2
        %is the y position
%OUTPUT:
        %angles: n,2 array the angle conversion of the points givin in the
        %input
function [angles] = convertXY2Angles(points)
x2 = points(:,1);
y2 = points(:,2);


theta2 = atand(1./(cosd(atand(x2)).*(y2)));
%correcting for oddities from matlabs atand function (converts to a
%specific range that isn't compatible
theta2(y2 < 0) = theta2(y2<0)+180;
% theta2 = theta2;

alpha2 = acosd(sqrt((x2.^2 .* sind(theta2).^2)./(1+x2.^2)));

%Shifitng angles 90 degrees so that the center value is 0
xs2 = (90-alpha2);
ys2 = (90-theta2);
xs2(x2 < 0) = -xs2(x2<0);

angles = [xs2, ys2];

% ax = ((angles(:,1)-min(angles(:,1)))./range(angles(:,1))).*80 - 40;
% ay = ((angles(:,2)-min(angles(:,2)))./range(angles(:,2))).*80 - 40;
% angles(:,1) = ax;
% angles(:,2) = ay;


end
