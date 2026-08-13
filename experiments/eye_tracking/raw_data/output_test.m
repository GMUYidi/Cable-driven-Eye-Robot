clc;
clear;
addpath('C:\Users\Administrator\Desktop\GMU research\295128-F2-USB-220-x64-uEye\ViewPoint\Interfaces\3rdParty\Microsoft-Windows\MATLAB\ViewPoint_EyeTracker_Toolbox');
vpx_Initialize;

sampling_gap = 0.1;

datatime_cell = [];
deltadatatime_cell = [];
GazeAngle_cell = [];
GazeAngleSmoothed_cell = [];
GazePoint_cell = [];
GazePointSmoothed_cell = [];
GlintPoint_cell = [];
PupilAspectRatio_cell = [];
PupilPoint_cell = [];
PupilSize_cell = [];
Torsion_cell = [];


headposition_cell = [];
pause(1)
for i = 1:150
    pause(sampling_gap)
    eyetype_left = 0;
    eyetype_right = 1;
    
    [deltadatatime_L]=vpx_GetDataDeltaTime(eyetype_left);
    [deltadatatime_R]=vpx_GetDataDeltaTime(eyetype_right);
    deltadatatime_cell = [deltadatatime_cell; deltadatatime_L, deltadatatime_R];
    
    [datatime_L]=vpx_GetDataTime(eyetype_left);
    [datatime_R]=vpx_GetDataTime(eyetype_right);
    datatime_cell = [datatime_cell;datatime_L,datatime_R];
    
    [xangle_L, yangle_L]=vpx_GetGazeAngle(eyetype_left);
    [xangle_R, yangle_R]=vpx_GetGazeAngle(eyetype_right);
    GazeAngle_cell = [GazeAngle_cell;xangle_L, yangle_L, xangle_R, yangle_R];
    
    [xangle_L, yangle_L]=vpx_GetGazeAngleSmoothed(eyetype_left);
    [xangle_R, yangle_R]=vpx_GetGazeAngleSmoothed(eyetype_right);
    GazeAngleSmoothed_cell = [GazeAngleSmoothed_cell; xangle_L, yangle_L, xangle_R, yangle_R];
    
    [xposition_L, yposition_L]=vpx_GetGazePoint(eyetype_left);
    [xposition_R, yposition_R]=vpx_GetGazePoint(eyetype_right);
    GazePoint_cell = [GazePoint_cell; xposition_L, yposition_L, xposition_R, yposition_R];
    
    [xposition_L, yposition_L]=vpx_GetGazePointSmoothed(eyetype_left);
    [xposition_R, yposition_R]=vpx_GetGazePointSmoothed(eyetype_right);
    GazePointSmoothed_cell = [GazePointSmoothed_cell; xposition_L, yposition_L, xposition_R, yposition_R];
    
    [xposition_L, yposition_L]=vpx_GetGlintPoint(eyetype_left);
    [xposition_R, yposition_R]=vpx_GetGlintPoint(eyetype_right);
    GlintPoint_cell = [GlintPoint_cell; xposition_L, yposition_L, xposition_R, yposition_R];
    
    [xposition_L, yposition_L]=vpx_GetPupilPoint(eyetype_left);
    [xposition_R, yposition_R]=vpx_GetPupilPoint(eyetype_right);
    PupilPoint_cell = [PupilPoint_cell; xposition_L, yposition_L, xposition_R, yposition_R];
    
    [PupilAspectRatio_L]=vpx_GetPupilAspectRatio(eyetype_left);
    [PupilAspectRatio_R]=vpx_GetPupilAspectRatio(eyetype_right);
    PupilAspectRatio_cell = [PupilAspectRatio_cell; PupilAspectRatio_L, PupilAspectRatio_R];
    
    [xsize_L,ysize_L]=vpx_GetPupilSize(eyetype_left);
    [xsize_R,ysize_R]=vpx_GetPupilSize(eyetype_right);
    PupilSize_cell = [PupilSize_cell; xsize_L,ysize_L,xsize_R,ysize_R]; 
    
    %[Torsion_L]=vpx_GetTorsion(eyetype_left);
    %[Torsion_R]=vpx_GetTorsion(eyetype_right);
    %Torsion_cell(i) = [Torsion_L,Torsion_R];
    %[TotalVelocity_L]=vpx_GetTotalVelocity(eyetype_left);
    %[TotalVelocity_R]=vpx_GetTotalVelocity(eyetype_right);
    %TotalVelocity_cell(i) = [TotalVelocity_L,TotalVelocity_R];
       
    [posangle]=vpx_GetHeadPositionAngle();
    headposition_cell = [headposition_cell;posangle];
    
end
save('output_test_2024_0122(0.1).mat','datatime_cell','deltadatatime_cell','GazeAngle_cell','GazeAngleSmoothed_cell',...
    'GazePoint_cell','GazePointSmoothed_cell','GlintPoint_cell','PupilPoint_cell','PupilAspectRatio_cell',...
    'PupilSize_cell','headposition_cell');