from dynamixel_sdk import *  # Uses Dynamixel SDK library
import time
import numpy as np
import pygame
import pyautogui

# Control table address
ADDR_GOAL_CURRENT = 102  # Assuming address for goal current is 102
ADDR_TORQUE_ENABLE = 64  # Control table address is different in Dynamixel model
ADDR_PRESENT_POSITION = 132
# Operating Mode Value
CURRENT_CONTROL_MODE = 0  # Typically 0 for current control mode
EXTENDED_POSITION_MODE = 4

DXL_MINIMUM_POSITION_VALUE  = 0
DXL_MAXIMUM_POSITION_VALUE  = 4095
BAUDRATE                    = 57600
PROTOCOL_VERSION            = 2.0
DXL_ID_1                    = 1
DXL_ID_2                    = 2
DXL_ID_3                    = 3
DXL_ID_4                    = 4
DXL_ID_5                    = 5
DXL_ID_6                    = 6
DXL_ID_7                    = 7
DXL_ID_8                    = 8
DXL_ID_9                    = 9
DXL_ID_10                   = 10
DXL_ID_11                   = 11
DXL_ID_12                   = 12
DXL_IDs = range(1, 13)
DEVICENAME                  = "COM14"

TORQUE_ENABLE               = 1     # Value for enabling the torque
TORQUE_DISABLE              = 0     # Value for disabling the torque
DXL_MOVING_STATUS_THRESHOLD = 20    # Dynamixel moving status threshold


# Control table address
ADDR_OPERATING_MODE = 11  # Assuming address for operating mode is 11
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132
ADDR_GOAL_VELOCITY = 104  # Assuming address for goal current is 102
ADDR_TORQUE_ENABLE = 64  # Control table address is different in Dynamixel model


# Initialize PortHandler instance
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if portHandler.openPort():
    print("Succeeded to open the port")
else:
    print("Failed to open the port")
    quit()

# Set port baudrate
if portHandler.setBaudRate(BAUDRATE):
    print("Succeeded to change the baudrate")
else:
    print("Failed to change the baudrate")
    quit()

pygame.init()

pygame.joystick.init()

if pygame.joystick.get_count() > 0:
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Joystick detected: {joystick.get_name()}")
else:
    print("No joystick detected!")
    pygame.quit()
    exit()

# Enable Dynamixel Torque and set to Extended Position Mode
for dxl_id in DXL_IDs:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OPERATING_MODE, EXTENDED_POSITION_MODE)
    if dxl_comm_result == COMM_SUCCESS:
        print("Servo ID {} has been set to Extended Position Mode.".format(dxl_id))
    else:
        print("Failed to set servo ID {}. Error code: {}".format(dxl_id, dxl_comm_result))




def enable_torque(enable=True):
    value = 1 if enable else 0
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_1, ADDR_TORQUE_ENABLE, value)
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_2, ADDR_TORQUE_ENABLE, value)
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_3, ADDR_TORQUE_ENABLE, value)
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_4, ADDR_TORQUE_ENABLE, value)
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_5, ADDR_TORQUE_ENABLE, value)
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_6, ADDR_TORQUE_ENABLE, value)
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_7, ADDR_TORQUE_ENABLE, value)
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_8, ADDR_TORQUE_ENABLE, value)
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_9, ADDR_TORQUE_ENABLE, value)
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_10, ADDR_TORQUE_ENABLE, value)
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_11, ADDR_TORQUE_ENABLE, value)
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID_12, ADDR_TORQUE_ENABLE, value)
    if dxl_comm_result != COMM_SUCCESS:
        print("写入失败：%s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("错误：%s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("舵机扭矩已成功%s" % ("启用" if enable else "禁用"))
        
# Function to set goal current based on desired angle
def set_goal_position(DXL_id,position):  
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_id, ADDR_GOAL_POSITION, position)
    

            
def fix_angle(dxl_present_position_a):
    if dxl_present_position_a > 2**31:
        dxl_present_position_a -= 2**32
    return dxl_present_position_a
            
def set_goal_degree(DXL_id_a,position):
    dxl_present_position_a, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_id_a, ADDR_PRESENT_POSITION)
    dxl_present_position_a = fix_angle(dxl_present_position_a)
    #print(dxl_present_position_a)
    position = int(position)
    set_goal_position(DXL_id_a,position)

def reset_initial():
    dxl_calibration_position_1, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, 1, ADDR_PRESENT_POSITION)
    dxl_calibration_position_2, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, 2, ADDR_PRESENT_POSITION)
    dxl_calibration_position_3, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, 3, ADDR_PRESENT_POSITION)
    dxl_calibration_position_4, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, 4, ADDR_PRESENT_POSITION)
    dxl_calibration_position_5, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, 5, ADDR_PRESENT_POSITION)
    dxl_calibration_position_6, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, 6, ADDR_PRESENT_POSITION)
    dxl_calibration_position_7, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, 7, ADDR_PRESENT_POSITION)
    dxl_calibration_position_8, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, 8, ADDR_PRESENT_POSITION)
    dxl_calibration_position_9, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, 9, ADDR_PRESENT_POSITION)
    dxl_calibration_position_10, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, 10, ADDR_PRESENT_POSITION)
    dxl_calibration_position_11, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, 11, ADDR_PRESENT_POSITION)
    dxl_calibration_position_12, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, 12, ADDR_PRESENT_POSITION)
    
    set_goal_degree(1,dxl_initial_position_1)
    set_goal_degree(2,dxl_initial_position_2)
    set_goal_degree(3,dxl_initial_position_3)
    set_goal_degree(4,dxl_initial_position_4)
    set_goal_degree(5,dxl_initial_position_5)
    set_goal_degree(6,dxl_initial_position_6)
    set_goal_degree(7,dxl_initial_position_7)
    set_goal_degree(8,dxl_initial_position_8)
    set_goal_degree(9,dxl_initial_position_9)
    set_goal_degree(10,dxl_initial_position_10)
    set_goal_degree(11,dxl_initial_position_11)
    set_goal_degree(12,dxl_initial_position_12)
    
    
        
def set_all_goal_degree(diff_1,diff_2,diff_3,diff_4,diff_5,diff_6,diff_7,diff_8,diff_9,diff_10,diff_11,diff_12):
    dxl_target_position_1 = dxl_initial_position_1+diff_1
    dxl_target_position_2 = dxl_initial_position_2+diff_2
    dxl_target_position_3 = dxl_initial_position_3+diff_3
    dxl_target_position_4 = dxl_initial_position_4+diff_4
    dxl_target_position_5 = dxl_initial_position_5+diff_5
    dxl_target_position_6 = dxl_initial_position_6+diff_6
    dxl_target_position_7 = dxl_initial_position_7+diff_7
    dxl_target_position_8 = dxl_initial_position_8+diff_8
    dxl_target_position_9 = dxl_initial_position_9+diff_9
    dxl_target_position_10 = dxl_initial_position_10+diff_10
    dxl_target_position_11 = dxl_initial_position_11+diff_11
    dxl_target_position_12 = dxl_initial_position_12+diff_12
        
    set_goal_degree(1,dxl_target_position_1)
    set_goal_degree(2,dxl_target_position_2)
    set_goal_degree(3,dxl_target_position_3)
    set_goal_degree(4,dxl_target_position_4)
    set_goal_degree(5,dxl_target_position_5)
    set_goal_degree(6,dxl_target_position_6)
    set_goal_degree(7,dxl_target_position_7)
    set_goal_degree(8,dxl_target_position_8)
    set_goal_degree(9,dxl_target_position_9)
    set_goal_degree(10,dxl_target_position_10)
    set_goal_degree(11,dxl_target_position_11)
    set_goal_degree(12,dxl_target_position_12)

        
def calc_angle_left(x,y,z):

    rate = 20
    radius = 14
    
    SR = np.array([-6, 0, 12.5])
    IR = np.array([-6, 0, -12.5])
    S_SR = np.array([-21, 0, 5])
    S_IR = np.array([-21, 0, -5])
    SO = np.array([-6, 0, 12.5])
    IO = np.array([-6, 0, -12.5])
    S_SO = np.array([12, 24, 7])
    S_IO = np.array([12, 24, -7])
    MR = np.array([0, radius, 0])
    LR = np.array([0, -radius, 0])
    S_MR = np.array([-21, 4.5, 0])
    S_LR = np.array([-21, -4.5, 0])
    
    yaw = np.radians(x)
    pitch = np.radians(y)
    roll = np.radians(z)
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    R = Rz @ (Ry @ Rx)
    
    R_SO = R @ SO
    R_IO = R @ IO
    R_SR = R @ SR
    R_IR = R @ IR
    R_MR = R @ MR
    R_LR = R @ LR
    
    SR_R_differ = np.linalg.norm(SR - S_SR) - np.linalg.norm(R_SR - S_SR)
    IR_R_differ = np.linalg.norm(IR - S_IR) - np.linalg.norm(R_IR - S_IR)
    SO_R_differ = np.linalg.norm(SO - S_SO) - np.linalg.norm(R_SO - S_SO)
    IO_R_differ = np.linalg.norm(IO - S_IO) - np.linalg.norm(R_IO - S_IO)
    MR_R_differ = np.linalg.norm(MR - S_MR) - np.linalg.norm(R_MR - S_MR)
    LR_R_differ = np.linalg.norm(LR - S_LR) - np.linalg.norm(R_LR - S_LR)
    
    SR_R_pulse = SR_R_differ / rate * 4096* 1.6
    IR_R_pulse = IR_R_differ / rate * 4096* 1.6
    SO_R_pulse = SO_R_differ / rate * 4096* 2 - 50
    IO_R_pulse = -IO_R_differ / rate * 4096* 2 - 50
    MR_R_pulse = MR_R_differ / rate * 4096* 1.8
    LR_R_pulse = LR_R_differ / rate * 4096* 1.8
    
    return MR_R_pulse, LR_R_pulse, SR_R_pulse, IR_R_pulse, SO_R_pulse, IO_R_pulse

def calc_angle_right(x,y,z):

    rate = 20
    radius = 14
    
    SR = np.array([-6, 0, 12.5])
    IR = np.array([-6, 0, -12.5])
    S_SR = np.array([-21, 0, 5])
    S_IR = np.array([-21, 0, -5])
    SO = np.array([-6, 0, 12.5])
    IO = np.array([-6, 0, -12.5])
    S_SO = np.array([12, -24, 7])
    S_IO = np.array([12, -24, -7])
    MR = np.array([0, radius, 0])
    LR = np.array([0, -radius, 0])
    S_MR = np.array([-21, -4.5, 0])
    S_LR = np.array([-21, 4.5, 0])
    
    yaw = np.radians(x)
    pitch = np.radians(y)
    roll = np.radians(z)
    
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw), np.cos(yaw), 0],
                   [0, 0, 1]])
    
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0, 1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll), np.cos(roll)]])
    
    R = Rz @ (Ry @ Rx)
    
    R_SO = R @ SO
    R_IO = R @ IO
    R_SR = R @ SR
    R_IR = R @ IR
    R_MR = R @ MR
    R_LR = R @ LR
    
    SR_R_differ = np.linalg.norm(SR - S_SR) - np.linalg.norm(R_SR - S_SR)
    IR_R_differ = np.linalg.norm(IR - S_IR) - np.linalg.norm(R_IR - S_IR)
    SO_R_differ = np.linalg.norm(SO - S_SO) - np.linalg.norm(R_SO - S_SO)
    IO_R_differ = np.linalg.norm(IO - S_IO) - np.linalg.norm(R_IO - S_IO)
    MR_R_differ = np.linalg.norm(MR - S_MR) - np.linalg.norm(R_MR - S_MR)
    LR_R_differ = np.linalg.norm(LR - S_LR) - np.linalg.norm(R_LR - S_LR)
    
    SR_R_pulse = -SR_R_differ / rate * 4096* 1.8
    IR_R_pulse = -IR_R_differ / rate * 4096* 1.8
    SO_R_pulse = -SO_R_differ / rate * 4096* 2 - 50
    IO_R_pulse = IO_R_differ / rate * 4096* 2 - 50
    MR_R_pulse = -MR_R_differ / rate * 4096* 2.0
    LR_R_pulse = -LR_R_differ / rate * 4096* 2.0
    
    return MR_R_pulse, LR_R_pulse, SR_R_pulse, IR_R_pulse, SO_R_pulse, IO_R_pulse

# Set goal current for a specific angle
enable_torque(True)
dxl_initial_position_1 = 948
dxl_initial_position_2 = 3012
dxl_initial_position_3 = 2640
dxl_initial_position_4 = 1544
dxl_initial_position_5 = 1690
dxl_initial_position_6 = 1724
dxl_initial_position_7 = 1530
dxl_initial_position_8 = 2904
dxl_initial_position_9 = 1180
dxl_initial_position_10 = 3770
dxl_initial_position_11 = 3102
dxl_initial_position_12 = 1412

reset_initial()

try:
    while True:
        x_axis, y_axis = pyautogui.position()
        screenWidth, screenHeight = pyautogui.size()
        if abs(x_axis) > screenWidth or abs(y_axis) > screenHeight:
            x_axis = 0
            y_axis = 0
        x = -x_axis/screenWidth*24+12
        y = -y_axis/screenHeight*18+9
        z = np.arcsin((np.sin(np.radians(x)) * np.sin(np.radians(y))) / (1 + (np.cos(np.radians(x)) * np.cos(np.radians(y))))) * np.degrees(1)
        print(x, y, z)
        MR_R_pulse_l, LR_R_pulse_l, SR_R_pulse_l, IR_R_pulse_l, SO_R_pulse_l, IO_R_pulse_l = calc_angle_left(x, y, z)
        MR_R_pulse_r, LR_R_pulse_r, SR_R_pulse_r, IR_R_pulse_r, SO_R_pulse_r, IO_R_pulse_r = calc_angle_right(x, y, z)
        if any(abs(val) > 9192 for val in [MR_R_pulse_l, LR_R_pulse_l, SR_R_pulse_l, IR_R_pulse_l, SO_R_pulse_l, IO_R_pulse_l, MR_R_pulse_r, LR_R_pulse_r, SR_R_pulse_r, IR_R_pulse_r, SO_R_pulse_r, IO_R_pulse_r]):
            # If any value exceeds the threshold, set all to 0
            MR_R_pulse_l = LR_R_pulse_l = SR_R_pulse_l = IR_R_pulse_l = SO_R_pulse_l = IO_R_pulse_l = MR_R_pulse_r = LR_R_pulse_r = SR_R_pulse_r = IR_R_pulse_r = SO_R_pulse_r = IO_R_pulse_r = 0

        set_all_goal_degree(MR_R_pulse_l, LR_R_pulse_l, SR_R_pulse_l, IR_R_pulse_l, SO_R_pulse_l, IO_R_pulse_l, MR_R_pulse_r, LR_R_pulse_r, SR_R_pulse_r, IR_R_pulse_r, SO_R_pulse_r, IO_R_pulse_r)
except Exception as e:
    print("程序被用户中断，错误信息：", e)
finally:
    reset_initial()  
    enable_torque(False)  
    portHandler.closePort()  
    print("finish and reset")
