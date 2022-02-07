import cv2
import numpy as np
import time
from dronekit import connect, VehicleMode


# ls -l /dev

#vehicle = connect('/dev/ttyS0', wait_ready=True, baud=921600)
vehicle = connect('/dev/ttyS0', baud=921600)
#vehicle = connect('/dev/ttyAMA0', baud=921600)

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

"""
#Capturing Real Time Video
cap = cv2.VideoCapture(0)
#width
#cap.set(3,432)
#height
#cap.set(4,432)

#ORB
orb = cv2.ORB_create()

#for ORB

indexparameters= dict(algorithm = 6,
                     table_number = 12,#6, # 12
                     key_size = 20,#12,     # 20
                     multi_probe_level = 2)#1) #2

searchparameters = dict(checks=30)
flann = cv2.FlannBasedMatcher(indexparameters, searchparameters)
if cap.isOpened():
    ret , frame = cap.read()
    #cap.read() returns a bool (True/False). If frame is read correctly, it will be True. So you can check end of the video by checking this return value.
    #ret will store that bool value 

else:
    ret = False

while ret:
    #cv2.imshow("ORB-TARGET", output_orb)

    ret , frame = cap.read()
    medianBlur = cv2.medianBlur(frame,5)
    grayFrame = cv2.cvtColor(medianBlur,cv2.COLOR_BGR2GRAY)
    #ORB ALGORITHM APPLIED TO REAL TIME CAPTURING
    keypoints_grayFrame_orb , descriptors_grayFrame_orb = orb.detectAndCompute(grayFrame, None)
    show_keypoints_grayFrame_orb = cv2.drawKeypoints(grayFrame,keypoints_grayFrame_orb, None)
    cv2.imshow("Real Time Cap orb", show_keypoints_grayFrame_orb)

    if cv2.waitKey(1) == 27:
            break

# When everything done, release the capture
cv2.destroyAllWindows()
cap.release()
"""
