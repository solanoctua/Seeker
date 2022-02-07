#-*- coding: utf-8 -*-
import cv2, math
import numpy as np

import time
from dronekit import connect, VehicleMode, LocationGlobal, LocationGlobalRelative
from pymavlink import mavutil # Needed for command message definitions
from ArucoSingleTracker import *


def uav_to_ne(x_obj, y_obj, yaw_rad_uav):
    #Two right-handed variants exist: east, north, up (ENU) coordinates and north, east, down (NED) coordinates.
    # takes x,y location of some object wrt uav and uav's yaw value, returns north and east location of the object
    c = math.cos(yaw_rad_uav)
    s = math.sin(yaw_rad_uav)
    
    north = x_obj*c - y_obj*s
    east = x_obj*s + y_obj*c 
    return(north, east)

def marker_position_to_angle(x, y, z):
    
    angle_x = math.atan2(x,z)
    angle_y = math.atan2(y,z)
    #print("angle_x, angle_y: ",angle_x, angle_y)
    return (angle_x, angle_y)

def check_angle_descend(angle_x, angle_y, angle_desc):
    angle_obj = math.sqrt(angle_x**2 + angle_y**2)
    print("IS angle_obj= {} <= {} = angle_desc".format(angle_obj,angle_desc))
    return(angle_obj <= angle_desc)

def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a Location object containing the latitude/longitude `dNorth` and `dEast` metres from the
    specified `original_location`. The returned Location has the same `alt and `is_relative` values
    as `original_location`.
    The function is useful when you want to move the vehicle around specifying locations relative to
    the current vehicle position.
    The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.
    For more information see:
    http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
    """
    earth_radius = 6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))
    
    print("dlat, dlon", dLat, dLon)

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    return(newlat, newlon)

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
        vehicle.mode    = VehicleMode("GUIDED")
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
def get_location_metres(original_location, dNorth, dEast):
    """
    Returns a LocationGlobal object containing the latitude/longitude `dNorth` and `dEast` metres from the
    specified `original_location`. The returned LocationGlobal has the same `alt` value
    as `original_location`.

    The function is useful when you want to move the vehicle around specifying locations relative to
    the current vehicle position.

    The algorithm is relatively accurate over small distances (10m within 1km) except close to the poles.

    For more information see:
    http://gis.stackexchange.com/questions/2951/algorithm-for-offsetting-a-latitude-longitude-by-some-amount-of-meters
    """
    earth_radius=6378137.0 #Radius of "spherical" earth
    #Coordinate offsets in radians
    dLat = dNorth/earth_radius
    dLon = dEast/(earth_radius*math.cos(math.pi*original_location.lat/180))

    #New position in decimal degrees
    newlat = original_location.lat + (dLat * 180/math.pi)
    newlon = original_location.lon + (dLon * 180/math.pi)
    if type(original_location) is LocationGlobal:
        targetlocation=LocationGlobal(newlat, newlon,original_location.alt)
    elif type(original_location) is LocationGlobalRelative:
        targetlocation = LocationGlobalRelative(newlat, newlon,original_location.alt)
    else:
        raise Exception("Invalid Location object passed")

    return targetlocation

"""
cam_calib_path = ""
camera_matrix       = np.loadtxt(calib_path+'cameraMatrix_raspi.txt', delimiter=',')
dist_coeffs   = np.loadtxt(calib_path+'cameraDistortion_raspi.txt', delimiter=',')
"""
camera_matrix  = np.array( [ [ 6.0784457614025803e+02          ,            0.         ,   3.2306280294111639e+02],
                                 [           0.                    , 6.0776688471513398e+02,   2.3031092286690938e+02],
                                 [           0.                    ,            0.         ,             1.          ]])
dist_coeffs  = np.array( [ 2.2848729690411801e-01, -9.5226223066766025e-01, 9.7446890329030999e-04, 8.5096039022205565e-05, 1.3294917618602062e-01 ])                                     
    
aruco_tracker = ArucoSingleTracker(marker_id=4, marker_size=0.175, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

rad_2_deg   = 180.0/math.pi
deg_2_rad   = 1.0/rad_2_deg 
#Connect Samurai
#vehicle = connect('/dev/ttyS0', baud=921600)
vehicle = connect('/dev/ttyAMA0', baud=921600)
"""
#Connect SITL
 run this on cmd 
 cd AppData\Local\Programs\Python\Python36-32\Scripts
 dronekit-sitl copter
"""
#vehicle = connect('tcp:127.0.0.1:5762', wait_ready=True)

# Get some vehicle attributes (state)
print("Get some vehicle attribute values:")
print("GPS: %s" % vehicle.gps_0)
print("Local Location: %s" % vehicle.location.local_frame)
print("Global Location (relative altitude): %s" % vehicle.location.global_relative_frame)
print("Battery: %s" % vehicle.battery)
print("Attitude: %s" % vehicle.attitude)
print("Heading: %s" % vehicle.heading)
print("Velocity: %s" % vehicle.velocity)
print("Last Heartbeat: %s" % vehicle.last_heartbeat)
print("Is Armable?: %s" % vehicle.is_armable)
print("System status: %s" % vehicle.system_status.state)
print("Mode: %s" % vehicle.mode.name)    # settable

vehicle.mode    = VehicleMode("GUIDED")
print("Starting the mission..")
print("Mode: %s" % vehicle.mode.name) 

vehicle.airspeed=0.20 # meters
landing_altitude = 0.90 # meters
angle_descend = 20
land_speed = 0.30 # meters
arm_and_takeoff(2) # altitude = 2 meters
time.sleep(2)
#vehicle.flush()
vehicle.mode    = VehicleMode("LOITER")
print("Mode: %s" % vehicle.mode.name)
#time.sleep(1)
time_0 = time.time()
uav_anchor = vehicle.location.global_relative_frame
print("uav_location (anchor point):",uav_anchor)
while True :
    uav_location = vehicle.location.global_relative_frame
    #print("uav_location: ",uav_location)
    marker_found, x_marker,y_marker,z_marker,roll_camera,pitch_camera = aruco_tracker.Seek(is_loop = False)
    if marker_found:
        #print("marker_found, x_marker,y_marker,z_marker,roll_camera,pitch_camera")
        #print(marker_found, x_marker,y_marker,z_marker,roll_camera,pitch_camera)
        # If uav altitude is high,do not trust aruco_tracker use barometer value for z_marker 
        if uav_location.alt >= 5.0:
            print("uav altitude >= 5.0 meters")
            z_marker = uav_location.alt
        #print("x_marker,y_marker,z_marker: ",x_marker,y_marker,z_marker)
        #angle_x, angle_y    = marker_position_to_angle(x_marker, y_marker, z_marker)
        if (time.time() >= time_0 + 1.0):
            time_0 = time.time()

            marker_north, marker_east  = uav_to_ne(x_marker, y_marker, vehicle.attitude.yaw)
            #print("marker_north, marker_east: ",marker_north, marker_east)
            marker_lon_lat  = get_location_metres(uav_location, marker_north*0.01, marker_east*0.01)
            #print(marker_lon_lat)
            marker_lon = get_location_metres(uav_location, marker_north, marker_east).lon
            marker_lat = get_location_metres(uav_location, marker_north, marker_east).lat
            #print("marker_lat, marker_lon: ",marker_lat, marker_lon)
            """
            if check_angle_descend(angle_x, angle_y, angle_descend):
                print("descending...")
                location_marker = LocationGlobalRelative(marker_lat, marker_lon, uav_location.alt - land_speed)
            else:
                print("closing for descend...")
                location_marker  = LocationGlobalRelative(marker_lat, marker_lon, uav_location.alt)
            """
            if (abs(roll_camera) <= angle_descend and abs(pitch_camera) <= angle_descend):
                print("Descending...")
                location_marker = LocationGlobalRelative(marker_lat, marker_lon, uav_location.alt - land_speed)
            else:
                print("Closing for descend...")
                location_marker  = LocationGlobalRelative(marker_lat, marker_lon, uav_location.alt)

            vehicle.simple_goto(location_marker)
            print("UAV Location    Lat = %.7f  Lon = %.7f  Alt = %.3f"%(uav_location.lat, uav_location.lon, uav_location.alt))
            print("Commanding to   Lat = %.7f  Lon = %.7f  Alt = %.3f"%(location_marker.lat, location_marker.lon, location_marker.alt))
            
            if (uav_location.alt <= landing_altitude):
                
                if (vehicle.mode.name != "LAND"):
                    vehicle.mode = VehicleMode("LAND")
                    time.sleep(1)
                    print("Vehicle Closing...")
                    vehicle.close()
                    break
    else:
        print("Marker not found, returning to anchor point")
        vehicle.simple_goto(uav_anchor)

