import cv2
import numpy as np
import time
import sys
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

# Get all channel values from RC transmitter
print ("Channel values from RC Tx:", vehicle.channels)

# Access channels individually
print ("Read channels individually:")
print (" Ch1: %s" % vehicle.channels['1'])
"""
# Set Ch2 override to 200 using indexing syntax
vehicle.channels.overrides['2'] = 200
# Set Ch3, Ch4 override to 300,400 using dictionary syntax"
vehicle.channels.overrides = {'3':300, '4':400}
"""
#Capturing Real Time Video
cap = cv2.VideoCapture(0)
#width
#cap.set(3,432)
#height
#cap.set(4,432)

if cap.isOpened():
    ret , frame = cap.read()
    #cap.read() returns a bool (True/False). If frame is read correctly, it will be True. So you can check end of the video by checking this return value.
    #ret will store that bool value 
else:
    ret = False

while ret:
    _, frame = cap.read()
    cv2.imshow("Real Time", frame)

    if cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()
cap.release()
sys.exit("Quit")

