import cv2, time, math
import numpy as np
from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil # Needed for command message definitions

cam = cv2.VideoCapture(2)
frame_width, frame_height = (1280,720)  #1280,720
target_lock_radius = 75

def arm_and_takeoff(aTargetAltitude):
    #Arms vehicle and fly to aTargetAltitude.
    print("Basic pre-arm checks")
    # Don't try to arm until autopilot is ready
    while not vehicle.is_armable:
        print("Waiting for vehicle to initialise...")
        time.sleep(1)

    print("Arming motors")
    time.sleep(1)
    # Copter should arm in GUIDED mode
    vehicle.mode    = VehicleMode("GUIDED")
    print("Mode: %s" % vehicle.mode.name)
    vehicle.armed   = True

    while (not vehicle.mode.name=="GUIDED"  ):
        print("Getting ready to take off ...")
        vehicle.mode = VehicleMode("GUIDED")
        time.sleep(1)
    # Confirm vehicle armed before attempting to take off
    while not vehicle.armed:
        print("Waiting for arming...")
        time.sleep(1)   
   
    print("Taking off!")
     # Take off to target altitude
    vehicle.simple_takeoff(aTargetAltitude)
    while True:
        print("Altitude: %s" % vehicle.location.global_relative_frame.alt)
        print("Velocity: %s" % vehicle.velocity)
        # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command after Vehicle.simple_takeoff will execute immediately).
        
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95:
        #if wait_for_alt(alt = 1, epsilon=0.3, rel=True, timeout=None)
            print("Reached target altitude")
            break
        time.sleep(1)

def condition_yaw(heading, relative=False):
    if relative:
        is_relative = 1 #yaw relative to direction of travel
    else:
        is_relative = 0 #yaw is an absolute angle
    # create the CONDITION_YAW command using command_long_encode()
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, #command
        0, #confirmation
        heading,    # param 1, yaw in degrees
        0,          # param 2, yaw speed deg/s
        1,          # param 3, direction -1 ccw, 1 cw
        is_relative, # param 4, relative offset 1, absolute angle 0
        0, 0, 0)    # param 5 ~ 7 not used
    # send command to vehicle
    vehicle.send_mavlink(msg)

def send_body_ned_velocity(velocity_x, velocity_y, velocity_z, duration=0):
    """
    Body Fixed Frame (Attached to the aircraft):
    The x axis points in forward (defined by geometry and not by movement) direction. (= roll axis)
    The y axis points to the right (geometrically) (= pitch axis)
    The z axis points downwards (geometrically) (= yaw axis)
    
    NED Coordinate System:
    velocity_x > 0 => fly North
    velocity_x < 0 => fly South
    velocity_y > 0 => fly East
    velocity_y < 0 => fly West
    velocity_z < 0 => ascend
    velocity_z > 0 => descend 
    """
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED,  # frame Needs to be MAV_FRAME_BODY_NED for forward/back left/right control.
        0b0000111111000111, # type_mask
        0, 0, 0, # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z, # m/s
        0, 0, 0, # x, y, z acceleration
        0, 0)
    for x in range(0, duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)

time.sleep(30)
#vehicle = connect('/dev/ttyS0', baud=921600)
vehicle = connect('/dev/ttyAMA0', baud=921600)
# Get some vehicle attributes (state)
print (" Get some vehicle attribute values:")
print (" GPS: %s" % vehicle.gps_0)
print (" Battery: %s" % vehicle.battery)
print (" Attitude: %s" % vehicle.attitude)
print (" Velocity: %s" % vehicle.velocity)
print (" Last Heartbeat: %s" % vehicle.last_heartbeat)
print (" Is Armable?: %s" % vehicle.is_armable)
print (" System status: %s" % vehicle.system_status.state)
print (" Mode: %s" % vehicle.mode.name)    # settable

# Get all channel values from RC transmitter
print ("Channel values from RC Tx:", vehicle.channels)

stage = 0
while True:
    if(vehicle.channels['6']) >= 1500:
        #print("starting mission...")
        prev_frame_time = 0
        new_frame_time = 0
        if cam.isOpened():
            ret,frame = cam.read()
            output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 1, (frame_width, frame_height)) #https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html
        else: 
            ret = False
        while ret :
            ret,frame = cam.read()
            frame = cv2.resize(frame,(frame_width, frame_height ))
            
            frame =cv2.flip(frame,-1)
            center_frame = (frame_width//2,frame_height//2)
            #Calculate FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            
            """
                FORWARD: Yaw 0 absolute (North)
                BACKWARD: Yaw 180 absolute (South)
                LEFT: Yaw 270 absolute (West)
                RIGHT: Yaw 90 absolute (East)
            """
            if stage == 0:
                aTargetAltitude = 1 # in meters
                mission = "arm_and_takeoff({})".format(aTargetAltitude)
                print(mission)
                arm_and_takeoff(aTargetAltitude)
                vehicle.mode = VehicleMode("LOITER")
                
            if stage == 1:
                angle = 180
                mission = "condition_yaw({})".format(angle)
                print(mission)
                condition_yaw(angle)
                
            if stage == 2:
                velocity_x = 0.1 # in meters
                velocity_y = 0  
                velocity_z = 0
                duration = 10
                mission = "send_body_ned_velocity({}, {}, {}, {})".format(velocity_x, velocity_y, velocity_z, duration)
                print(mission)
                send_body_ned_velocity(velocity_x, velocity_y, velocity_z, 10)
                
            if stage == 3:
                angle = 90
                mission = "condition_yaw({})".format(angle)
                print(mission)
                condition_yaw(angle)
                
            if stage == 4:
                velocity_x = 0.2 # in meters
                velocity_y = 0  
                velocity_z = 0
                duration = 5
                mission = "send_body_ned_velocity({}, {}, {}, {})".format(velocity_x, velocity_y, velocity_z, duration)
                print(mission)
                send_body_ned_velocity(velocity_x, velocity_y, velocity_z, 10)
                
            if stage == 5:
                mission = "LAND"
                print(mission)
                vehicle.mode = VehicleMode("LAND")
                #disarm(wait=True, timeout=None)
                vehicle.close()
                
            stage += 1
            cv2.circle(frame, (frame_width//2, frame_height//2), target_lock_radius, (0,255,0), 1)
            cv2.line(frame,(int(frame_width/2),0),(int(frame_width/2),int(frame_height)),(0,255,0),1) # vertical line
            cv2.line(frame,(0,int(frame_height/2)),(frame_width,int(frame_height/2)),(0,255,0),1) # horizontal line
            cv2.putText(frame,"FPS:{}".format(int(fps)),(15,15),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1,cv2.LINE_AA)#Displays fps
            cv2.putText(frame, mission, (35,35),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0),1,cv2.LINE_AA)
            cv2.imshow("realTimeCamera", frame)
            output.write(frame) 

            key=cv2.waitKey(1)
            if key==27:
                break
        cv2.destroyAllWindows()
        output.release()
        cam.release()
        vehicle.mode = VehicleMode("LAND")
        #disarm(wait=True, timeout=None)
        vehicle.close()
    else:
        print("Waiting for command")
        print ("Channel values from RC Tx:", vehicle.channels)
        print("Command Channel(6): ",vehicle.channels['6'])
        time.sleep(3)