import numpy as np
import cv2 

path_of_images = "C:/Users/asus/Desktop/Samurai-Seeker/FisheyeCalibration"
cap = cv2.VideoCapture(0)
frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
print("frame_width: ",frame_width)
print("frame_height: ",frame_height)
count = 0
if cap.isOpened():
    ret,frame = cap.read()
else:
    ret = False
while ret:
    ret , frame = cap.read()
    cv2.putText(frame,"press 'esc' to quit",(15,15), cv2.FONT_HERSHEY_SIMPLEX, .6,(0,0,255),1,cv2.LINE_AA) #displays some text
    cv2.putText(frame,"press 's' to save the image",(15,35), cv2.FONT_HERSHEY_SIMPLEX, .6,(255,0,0),1,cv2.LINE_AA) #displays some text 
    cv2.imshow("camera",frame)
    
    key = cv2.waitKey(1)
    if key == ord("+"):
        print("Saving image to ",path_of_images)
        cv2.imwrite("{}{}.jpg".format(path_of_images+"/",count), frame)
        count += 1
    if key == 27:   #press esc to quit
        break
cv2.destroyAllWindows()
cap.release()