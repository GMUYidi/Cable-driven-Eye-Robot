%load('C:\Users\Administrator\Desktop\GMU research\295128-F2-USB-220-x64-uEye\ViewPoint\Data\modified_output_Allisn_2024_0209_test5(0.01)to(0.1).mat');
load('modified_output_Kabir_2024_0209_test4(0.01)to(0.1).mat')
XL = GazePointSmoothed_cell(:,1);
YL = 1-GazePointSmoothed_cell(:,2);
XR = GazePointSmoothed_cell(:,3);
YR = 1-GazePointSmoothed_cell(:,4);

XL_angle = GazeAngleSmoothed_cell(:,1);
YR_angle = GazeAngleSmoothed_cell(:,2);
XR_angle = GazeAngleSmoothed_cell(:,3);
YL_angle = GazeAngleSmoothed_cell(:,4);
timeL = datatime_cell(:,1);
timeR = datatime_cell(:,2);
indexXL = 1:length(XL_angle);
%%
h = figure;
plot(XL, YL, 'r-o', XR, YR, 'b-+');
title('Eyes Track');
txt = {'red:left eye'; 'blue:right eye'};
text(0.35, 0.4, txt, 'FontSize', 20);
pbaspect([8 7 1]); % 设置横纵坐标比例为3:2
print(h, '-djpeg', 'eyetrack.jpg');



%%
h=figure;
plot(indexXL,XL_angle,'r-o',indexXL,XR_angle,'b-+');
title('Eyes angleX')
txt = {'red:light eye'; 'blue:right eye'};
text(400,-10,txt,'FontSize',20);
print(h,'-djpeg','eyeangleX.jpg');

%%
h=figure;
plot(indexXL,YL_angle,'r-o',indexXL,YR_angle,'b-+');
title('Eyes angleY')
txt = {'red:light eye'; 'blue:right eye'};
text(900,-20,txt,'FontSize',20);
print(h,'-djpeg','eyeangleY.jpg');

%%
st = 1;
%plot_eye_graph(XL,YL,XR,YR,st)
%plot_eye_angle(XL_angle,YL_angle,XR_angle,YR_angle,st)
save('modified_output_A1_2024_0209_test4(0.01)to(0.1).mat','datatime_cell','deltadatatime_cell','GazeAngle_cell','GazeAngleSmoothed_cell',...
'GazePoint_cell','GazePointSmoothed_cell','GlintPoint_cell','PupilPoint_cell','PupilAspectRatio_cell',...
'PupilSize_cell');