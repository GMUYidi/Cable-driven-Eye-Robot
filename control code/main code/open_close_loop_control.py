from dynamixel_sdk import *  # Uses Dynamixel SDK library
import time
import numpy as np
import pandas as pd
from picamera.array import PiRGBArray
from picamera2 import Picamera2
import scipy.io as scio
import socket

# Initialize left eye camera
camera_left = Picamera2(camera_num=0)
camera_left.configure(camera_left.create_still_configuration())
camera_task_left = False

# Initialize right eye camera
camera_right = Picamera2(camera_num=1)
camera_right.configure(camera_right.create_still_configuration())
camera_task_right = False

# Control table addresses
ADDR_GOAL_CURRENT = 102  # Address for goal current
ADDR_TORQUE_ENABLE = 64  # Address for torque enable
ADDR_PRESENT_POSITION = 132  # Address for present position
ADDR_PRESENT_PWM = 124  # Address for present PWM

# Operating mode values
CURRENT_CONTROL_MODE = 0  # Current control mode
EXTENDED_POSITION_MODE = 4  # Extended position control mode

# Define Dynamixel servo parameters and IDs
DXL_MINIMUM_POSITION_VALUE = 0
DXL_MAXIMUM_POSITION_VALUE = 4095
BAUDRATE = 57600
PROTOCOL_VERSION = 2.0
DXL_ID_1 = 1
DXL_ID_2 = 2
DXL_ID_3 = 3
DXL_ID_4 = 4
DXL_ID_5 = 5
DXL_ID_6 = 6
DXL_ID_7 = 7
DXL_ID_8 = 8
DXL_ID_9 = 9
DXL_ID_10 = 10
DXL_ID_11 = 11
DXL_ID_12 = 12
DXL_IDs = range(1, 13)
DEVICENAME = "COM3"

TORQUE_ENABLE = 1  # Enable torque
TORQUE_DISABLE = 0  # Disable torque
DXL_MOVING_STATUS_THRESHOLD = 20  # Moving status threshold for Dynamixel

# Additional control table addresses
ADDR_OPERATING_MODE = 11  # Address for operating mode
ADDR_GOAL_POSITION = 116  # Address for goal position
ADDR_GOAL_VELOCITY = 104  # Address for goal velocity

# Initialize PortHandler instance
portHandler = PortHandler(DEVICENAME)

# Initialize PacketHandler instance
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Set Raspberry Pi IP address and port
RASPBERRY_PI_IP = '192.168.137.89'  # Change to your Pi's IP
PORT = 12345
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((RASPBERRY_PI_IP, PORT))
print("Connected to Raspberry Pi IMU server!")

buffer = ""  # Buffer for socket data
# Initialize target position records
last_goal_position = {dxl_id: 0 for dxl_id in DXL_IDs}

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

# Set all servos to Extended Position Mode
for dxl_id in DXL_IDs:
    # Disable torque before setting the mode
    packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, dxl_id, ADDR_OPERATING_MODE, EXTENDED_POSITION_MODE)
    if dxl_comm_result == COMM_SUCCESS:
        print("Servo ID {} has been set to Extended Position Mode.".format(dxl_id))
    else:
        print("Failed to set servo ID {}. Error code: {}".format(dxl_id, dxl_comm_result))

# Setup PID parameters
last_xangle_left = 0
last_yangle_left = 0
last_zangle_left = 0
last_xangle_right = 0
last_yangle_right = 0
last_zangle_right = 0
lx_left = 0
ly_left = 0
lz_left = 0
lx_right = 0
ly_right = 0
lz_right = 0

kpx = 2.5
kpy = 2.5
kpz = 2.5
kdx = 0.5
kdy = 0.5
kdz = 0.5

# Define the index for the current set of eye angles, from 0 to 499 (total 500 sets)
i = 0

# Load gaze angle data
data = scio.loadmat('/home/pi/Desktop/robotic_eyes_platform/data/STRYidi/STRoutput_modified_output_Y_2024_0205_test2(0.01)to(0.04).mat') 
XR = -data['GazeAngleSmoothed_cell'][:, 0]
YR = -data['GazeAngleSmoothed_cell'][:, 1]
XL = -data['GazeAngleSmoothed_cell'][:, 2]
YL = -data['GazeAngleSmoothed_cell'][:, 3]

xtarget_left = XL[i]
ytarget_left = YL[i]
ztarget_left = np.degrees(np.arcsin((np.sin(np.radians(xtarget_left)) * np.sin(np.radians(ytarget_left))) / (1 + (np.cos(np.radians(xtarget_left)) * np.cos(np.radians(ytarget_left))))))

xtarget_right = XL[i]
ytarget_right = YL[i]
ztarget_right = np.degrees(np.arcsin((np.sin(np.radians(xtarget_right)) * np.sin(np.radians(ytarget_right))) / (1 + (np.cos(np.radians(xtarget_right)) * np.cos(np.radians(ytarget_right))))))

# Update to the next target angles after the robotic eye reaches the current target
def repeat():
    global i
    global xtarget_left
    global ytarget_left
    global ztarget_left
    global xtarget_right
    global ytarget_right
    global ztarget_right
    global XL
    global YL
    global XR
    global YR

    if i < 500:
        i = i + 1
    xtarget_left = XL[i]
    ytarget_left = YL[i]
    xtarget_right = XR[i]
    ytarget_right = YR[i]
    ztarget_left = np.degrees(np.arcsin((np.sin(np.radians(xtarget_left)) * np.sin(np.radians(ytarget_left))) / (1 + (np.cos(np.radians(xtarget_left)) * np.cos(np.radians(ytarget_left))))))
    ztarget_right = np.degrees(np.arcsin((np.sin(np.radians(xtarget_right)) * np.sin(np.radians(ytarget_right))) / (1 + (np.cos(np.radians(xtarget_right)) * np.cos(np.radians(ytarget_right))))))

# Check if both robotic eyes reach the target angles within the tolerance range.
# If yes, capture the current images from cameras. Once both sides are captured, proceed to repeat.
def takeimage(x_left, y_left, z_left, x_right, y_right, z_right, xtarget_left, ytarget_left, ztarget_left, xtarget_right, ytarget_right, ztarget_right):
    if abs(x_left - xtarget_left) < 0.25 and abs(y_left - ytarget_left) < 0.25 and abs(z_left - ztarget_left) < 0.25:
        camera_left.start()
        camera_left.capture_file("/home/pi/Desktop/robotic_eyes_platform/data/left_image/" + 'STRnormal' + '/' + 'chessboard%s.jpg' % i)
        time.sleep(0.1)
        camera_left.stop()
        camera_task_left = True
    if abs(x_right - xtarget_right) < 0.25 and abs(y_right - ytarget_right) < 0.25 and abs(z_right - ztarget_right) < 0.25:
        camera_right.start()
        camera_right.capture_file("/home/pi/Desktop/robotic_eyes_platform/data/left_image/" + 'STRnormal' + '/' + 'chessboard%s.jpg' % i)
        time.sleep(0.1)
        camera_right.stop()
        camera_task_right = True
    if camera_task_right and camera_task_left:
        camera_task_left = False
        camera_task_right = False
        repeat()

# ---------------- IMU data reading function ------------------
# Read IMU data from Raspberry Pi using socket
def IMUsensor_data():
    global buffer
    while True:
        data = s.recv(1024).decode()
        buffer += data

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            if "error" in line:
                continue
            try:
                imu_1_str, imu_2_str = line.strip().split(";")
                yaw1, roll1, pitch1 = map(float, imu_1_str.split(","))
                yaw2, roll2, pitch2 = map(float, imu_2_str.split(","))
                if yaw1 > 180:
                    yaw1 -= 360
                if yaw2 > 180:
                    yaw2 -= 360
                print(yaw1, pitch1, roll1, yaw2, pitch2, roll2)
                return yaw1, pitch1, roll1, yaw2, pitch2, roll2
            except:
                continue

# Enable torque for all servos
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
        print("Write failed: %s" % packetHandler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("Error: %s" % packetHandler.getRxPacketError(dxl_error))
    else:
        print("Servo torque has been successfully %s" % ("enabled" if enable else "disabled"))

        
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

# Reset all servos to their initial positions
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

    # Move all servos to initial positions
    set_goal_degree(1, dxl_initial_position_1)
    set_goal_degree(2, dxl_initial_position_2)
    set_goal_degree(3, dxl_initial_position_3)
    set_goal_degree(4, dxl_initial_position_4)
    set_goal_degree(5, dxl_initial_position_5)
    set_goal_degree(6, dxl_initial_position_6)
    set_goal_degree(7, dxl_initial_position_7)
    set_goal_degree(8, dxl_initial_position_8)
    set_goal_degree(9, dxl_initial_position_9)
    set_goal_degree(10, dxl_initial_position_10)
    set_goal_degree(11, dxl_initial_position_11)
    set_goal_degree(12, dxl_initial_position_12)

# Control function to set the target position of all servos
def set_all_goal_degree(diff_1, diff_2, diff_3, diff_4, diff_5, diff_6, diff_7, diff_8, diff_9, diff_10, diff_11, diff_12):
    dxl_target_position_1 = dxl_initial_position_1 + diff_1
    dxl_target_position_2 = dxl_initial_position_2 + diff_2
    dxl_target_position_3 = dxl_initial_position_3 + diff_3
    dxl_target_position_4 = dxl_initial_position_4 + diff_4
    dxl_target_position_5 = dxl_initial_position_5 + diff_5
    dxl_target_position_6 = dxl_initial_position_6 + diff_6
    dxl_target_position_7 = dxl_initial_position_7 + diff_7
    dxl_target_position_8 = dxl_initial_position_8 + diff_8
    dxl_target_position_9 = dxl_initial_position_9 + diff_9
    dxl_target_position_10 = dxl_initial_position_10 + diff_10
    dxl_target_position_11 = dxl_initial_position_11 + diff_11
    dxl_target_position_12 = dxl_initial_position_12 + diff_12

    set_goal_degree(1, dxl_target_position_1)
    set_goal_degree(2, dxl_target_position_2)
    set_goal_degree(3, dxl_target_position_3)
    set_goal_degree(4, dxl_target_position_4)
    set_goal_degree(5, dxl_target_position_5)
    set_goal_degree(6, dxl_target_position_6)
    set_goal_degree(7, dxl_target_position_7)
    set_goal_degree(8, dxl_target_position_8)
    set_goal_degree(9, dxl_target_position_9)
    set_goal_degree(10, dxl_target_position_10)
    set_goal_degree(11, dxl_target_position_11)
    set_goal_degree(12, dxl_target_position_12)

# Open loop calculation for left eye.
# Calculate the required servo movements to reach the target gaze direction (x, y, z) from the current (real_yaw, real_pitch, real_roll)
def calc_angle_left(x, y, z, real_yaw, real_pitch, real_roll):

    rate = 20
    radius = 15

    # Define insertion points
    SR = np.array([-6, 0, 14])
    IR = np.array([-6, 0, -14])
    S_SR = np.array([-21, 0, 5])
    S_IR = np.array([-21, 0, -5])
    SO = np.array([-6, 0, 13.5])
    IO = np.array([-6, 0, -13.5])
    S_SO = np.array([12, 24, 7])
    S_IO = np.array([12, 24, -7])
    MR = np.array([0, radius, 0])
    LR = np.array([0, -radius, 0])
    S_MR = np.array([-21, 4.5, 0])
    S_LR = np.array([-21, -4.5, 0])

    # Compute rotation matrices
    yaw, pitch, roll = map(np.radians, [x, y, z])
    yawR, pitchR, rollR = map(np.radians, [real_yaw, real_pitch, real_roll])

    R = np.dot(np.dot(np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                [np.sin(yaw), np.cos(yaw), 0],
                                [0, 0, 1]]),
                      np.array([[np.cos(pitch), 0, np.sin(pitch)],
                                [0, 1, 0],
                                [-np.sin(pitch), 0, np.cos(pitch)]])),
               np.array([[1, 0, 0],
                         [0, np.cos(roll), -np.sin(roll)],
                         [0, np.sin(roll), np.cos(roll)]]))

    RR = np.dot(np.dot(np.array([[np.cos(yawR), -np.sin(yawR), 0],
                                 [np.sin(yawR), np.cos(yawR), 0],
                                 [0, 0, 1]]),
                       np.array([[np.cos(pitchR), 0, np.sin(pitchR)],
                                 [0, 1, 0],
                                 [-np.sin(pitchR), 0, np.cos(pitchR)]])),
                np.array([[1, 0, 0],
                          [0, np.cos(rollR), -np.sin(rollR)],
                          [0, np.sin(rollR), np.cos(rollR)]]))

    # Calculate transformed insertion points
    def transform_and_diff(point, fixed, R_current, R_target):
        return np.linalg.norm(R_current @ point - fixed) - np.linalg.norm(R_target @ point - fixed)

    SR_R_pulse = transform_and_diff(SR, S_SR, RR, R) / rate * 4096 * 3.5
    IR_R_pulse = transform_and_diff(IR, S_IR, RR, R) / rate * 4096 * 3.5
    SO_R_pulse = transform_and_diff(SO, S_SO, RR, R) / rate * 4096 * 2 - 50
    IO_R_pulse = -transform_and_diff(IO, S_IO, RR, R) / rate * 4096 * 2 - 50
    MR_R_pulse = transform_and_diff(MR, S_MR, RR, R) / rate * 4096 * 2.3
    LR_R_pulse = transform_and_diff(LR, S_LR, RR, R) / rate * 4096 * 2.3

    return MR_R_pulse, LR_R_pulse, SR_R_pulse, IR_R_pulse, SO_R_pulse, IO_R_pulse

# Same as above but for the right eye
def calc_angle_right(x, y, z, real_yaw, real_pitch, real_roll):

    rate = 15
    radius = 15

    # Define insertion points
    SR = np.array([-6, 0, 14])
    IR = np.array([-6, 0, -14])
    S_SR = np.array([-21, 0, 5])
    S_IR = np.array([-21, 0, -5])
    SO = np.array([-6, 0, 13.5])
    IO = np.array([-6, 0, -13.5])
    S_SO = np.array([12, -24, 7])
    S_IO = np.array([12, -24, -7])
    MR = np.array([0, radius, 0])
    LR = np.array([0, -radius, 0])
    S_MR = np.array([-21, -4.5, 0])
    S_LR = np.array([-21, 4.5, 0])

    yaw, pitch, roll = map(np.radians, [x, y, z])
    yawR, pitchR, rollR = map(np.radians, [real_yaw, real_pitch, real_roll])

    R = np.dot(np.dot(np.array([[np.cos(yaw), -np.sin(yaw), 0],
                                [np.sin(yaw), np.cos(yaw), 0],
                                [0, 0, 1]]),
                      np.array([[np.cos(pitch), 0, np.sin(pitch)],
                                [0, 1, 0],
                                [-np.sin(pitch), 0, np.cos(pitch)]])),
               np.array([[1, 0, 0],
                         [0, np.cos(roll), -np.sin(roll)],
                         [0, np.sin(roll), np.cos(roll)]]))

    RR = np.dot(np.dot(np.array([[np.cos(yawR), -np.sin(yawR), 0],
                                 [np.sin(yawR), np.cos(yawR), 0],
                                 [0, 0, 1]]),
                       np.array([[np.cos(pitchR), 0, np.sin(pitchR)],
                                 [0, 1, 0],
                                 [-np.sin(pitchR), 0, np.cos(pitchR)]])),
                np.array([[1, 0, 0],
                          [0, np.cos(rollR), -np.sin(rollR)],
                          [0, np.sin(rollR), np.cos(rollR)]]))

    def transform_and_diff(point, fixed, R_current, R_target):
        return np.linalg.norm(R_current @ point - fixed) - np.linalg.norm(R_target @ point - fixed)

    SR_R_pulse = -transform_and_diff(SR, S_SR, RR, R) / rate * 4096 * 2
    IR_R_pulse = -transform_and_diff(IR, S_IR, RR, R) / rate * 4096 * 2
    SO_R_pulse = -transform_and_diff(SO, S_SO, RR, R) / rate * 4096 * 2 - 50
    IO_R_pulse = transform_and_diff(IO, S_IO, RR, R) / rate * 4096 * 2 - 50
    MR_R_pulse = -transform_and_diff(MR, S_MR, RR, R) / rate * 4096 * 2.2
    LR_R_pulse = -transform_and_diff(LR, S_LR, RR, R) / rate * 4096 * 2.2

    return MR_R_pulse, LR_R_pulse, SR_R_pulse, IR_R_pulse, SO_R_pulse, IO_R_pulse


# Calculate PID parameters
def PIDcamera(x, y, z, lx, ly, lz, xtarget, ytarget, ztarget):
    threshold = 50
    pdx = kpx * (x - xtarget) + kdx * (x - lx)
    pdy = kpy * (y - ytarget) + kdy * (y - ly)
    pdz = kpz * (z - ztarget) + kdz * (z - lz)
    
    if abs(pdx) > threshold:
        pdx = np.sign(pdx) * threshold
    if abs(pdy) > threshold:
        pdy = np.sign(pdy) * threshold
    if abs(pdz) > threshold:
        pdz = np.sign(pdz) * threshold

    return pdx, pdy, pdz

# Fine adjustment function under closed-loop control
def angleturn(xtarget_left, ytarget_left, ztarget_left, xtarget_right, ytarget_right, ztarget_right):
    global lx_left, ly_left, lz_left, lx_right, ly_right, lz_right
    global last_xangle_left, last_yangle_left, last_zangle_left
    global last_xangle_right, last_yangle_right, last_zangle_right

    yaw_left, pitch_left, roll_left, yaw_right, pitch_right, roll_right = IMUsensor_data()
    takeimage(yaw_left, pitch_left, roll_left, yaw_right, pitch_right, roll_right, xtarget_left, ytarget_left, ztarget_left, xtarget_right, ytarget_right, ztarget_right)

    pdx_left, pdy_left, pdz_left = PIDcamera(yaw_left, pitch_left, roll_left, lx_left, ly_left, lz_left, xtarget_left, ytarget_left, ztarget_left)
    pdx_right, pdy_right, pdz_right = PIDcamera(yaw_right, pitch_right, roll_right, lx_right, ly_right, lz_right, xtarget_left, ytarget_left, ztarget_left)

    xangle_left = last_xangle_left - pdx_left
    yangle_left = last_yangle_left - pdy_left
    zangle_left = last_zangle_left - pdz_left
    xangle_right = last_xangle_right - pdx_right
    yangle_right = last_yangle_right - pdy_right
    zangle_right = last_zangle_right - pdz_right

    set_all_goal_degree(xangle_left, -xangle_left, -yangle_left, yangle_left, zangle_left, zangle_left,
                        -xangle_right, xangle_right, yangle_right, -yangle_right, -zangle_right, -zangle_right)

    last_xangle_left = xangle_left
    last_yangle_left = yangle_left
    last_zangle_left = zangle_left
    last_xangle_right = xangle_right
    last_yangle_right = yangle_right
    last_zangle_right = zangle_right

    lx_left = yaw_left
    ly_left = pitch_left
    lz_left = roll_left
    lx_right = yaw_right
    ly_right = pitch_right
    lz_right = roll_right

# Manually adjust all servos to the initial position so that both eyes are in the straight-ahead position
enable_torque(True)

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

dxl_initial_position_1 = dxl_calibration_position_1
dxl_initial_position_2 = dxl_calibration_position_2
dxl_initial_position_3 = dxl_calibration_position_3
dxl_initial_position_4 = dxl_calibration_position_4
dxl_initial_position_5 = dxl_calibration_position_5
dxl_initial_position_6 = dxl_calibration_position_6
dxl_initial_position_7 = dxl_calibration_position_7
dxl_initial_position_8 = dxl_calibration_position_8
dxl_initial_position_9 = dxl_calibration_position_9
dxl_initial_position_10 = dxl_calibration_position_10
dxl_initial_position_11 = dxl_calibration_position_11
dxl_initial_position_12 = dxl_calibration_position_12

reset_initial()

try:
    while True:
        # Execute open loop control strategy
        yaw_left, pitch_left, roll_left, yaw_right, pitch_right, roll_right = IMUsensor_data()
        MR_R_pulse_l, LR_R_pulse_l, SR_R_pulse_l, IR_R_pulse_l, SO_R_pulse_l, IO_R_pulse_l = calc_angle_left(xtarget_left, ytarget_left, ztarget_left, yaw_left, pitch_left, roll_left)
        MR_R_pulse_r, LR_R_pulse_r, SR_R_pulse_r, IR_R_pulse_r, SO_R_pulse_r, IO_R_pulse_r = calc_angle_right(xtarget_right, ytarget_right, ztarget_right, yaw_right, pitch_right, roll_right)

        set_all_goal_degree(MR_R_pulse_l, LR_R_pulse_l, SR_R_pulse_l, IR_R_pulse_l, SO_R_pulse_l, IO_R_pulse_l,
                            MR_R_pulse_r, LR_R_pulse_r, SR_R_pulse_r, IR_R_pulse_r, SO_R_pulse_r, IO_R_pulse_r)

        # Execute close loop control strategy
        angleturn(xtarget_left, ytarget_left, ztarget_left, xtarget_right, ytarget_right, ztarget_right)

        if i == 500:
            break

except Exception as e:
    print("Program interrupted, error message:", e)

finally:
    # Reset all servos and disable torque after completion
    # reset_initial()
    enable_torque(False)
    portHandler.closePort()
    s.close()
    print("Finish and reset completed")

