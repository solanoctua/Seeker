import cv2, time
import numpy as np
from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil # Needed for command message definitions

camera_matrix  = np.array( [ [ 6.0784457614025803e+02          ,            0.         ,   3.2306280294111639e+02],
                                 [           0.                    , 6.0776688471513398e+02,   2.3031092286690938e+02],
                                 [           0.                    ,            0.         ,             1.          ]])
dist_coeffs  = np.array( [ 2.2848729690411801e-01, -9.5226223066766025e-01, 9.7446890329030999e-04, 8.5096039022205565e-05, 1.3294917618602062e-01 ])

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
        # Wait until the vehicle reaches a safe height before processing the goto (otherwise the command
        #  after Vehicle.simple_takeoff will execute immediately).
        #Break and return from function just below target altitude.
        if vehicle.location.global_relative_frame.alt>=aTargetAltitude*0.95:
        #if wait_for_alt(alt = 1, epsilon=0.3, rel=True, timeout=None)
            print("Reached target altitude")
            break
        time.sleep(1)

def condition_yaw(heading, relative=False):
    if relative:
        is_relative=1 #yaw relative to direction of travel
    else:
        is_relative=0 #yaw is an absolute angle
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
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       # time_boot_ms (not used)
        0, 0,    # target system, target component
        mavutil.mavlink.MAV_FRAME_BODY_NED, # frame Needs to be MAV_FRAME_BODY_NED for forward/back left/right control.
        0b0000111111000111, # type_mask
        0, 0, 0, # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z, # m/s
        0, 0, 0, # x, y, z acceleration
        0, 0)
    for x in range(0,duration):
        vehicle.send_mavlink(msg)
        time.sleep(1)


time.sleep(20)
vehicle = connect('/dev/ttyS0', baud=921600)

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
print "Channel values from RC Tx:", vehicle.channels

if(vehicle.channels['?']) == ?:
    vehicle.airspeed=0.20 # meters
    landing_altitude = 0.90 # meters
    angle_descend = 20
    and_speed = 0.30 # meters
    vehicle.mode    = VehicleMode("GUIDED")
    print("Starting the mission..")
    print("Mode: %s" % vehicle.mode.name)
    arm_and_takeoff(2) # altitude = 2 meters
    time.sleep(2)
    while True :
        if Arrow:
            print("forward at 0.5 m/s for 5 sec.")
            # forward at 0.5 m/s for 5 sec.
            velocity_x = 0
            velocity_y = 0.5  
            velocity_z = 0
            duration = 5
            send_body_ned_velocity(velocity_x, velocity_y, velocity_z, duration)
        if Line:

        if T:
            vehicle.mode = VehicleMode("LAND")
            disarm(wait=True, timeout=None)
            vehicle.close()




else:
