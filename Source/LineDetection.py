import cv2, time, math
import numpy as np

def nothing(x):
    pass

cam = cv2.VideoCapture(0)

cv2.namedWindow("ColorTrackbars")
cv2.createTrackbar("min - H", "ColorTrackbars", 0, 179, nothing)
cv2.createTrackbar("min - S", "ColorTrackbars", 0, 255, nothing)
cv2.createTrackbar("min - V", "ColorTrackbars", 0, 255, nothing)
cv2.createTrackbar("max - H", "ColorTrackbars", 179, 179, nothing)
cv2.createTrackbar("max - S", "ColorTrackbars", 255, 255, nothing)
cv2.createTrackbar("max - V", "ColorTrackbars", 100, 255, nothing)

color = (255,255,255)
prev_frame_time = 0
new_frame_time = 0
if cam.isOpened():
    ret,frame = cam.read()
else: 
    ret = False
while ret :
    ret,frame = cam.read()
    
    frame = cv2.imread("Seeker/TargetImages/line1.png")
    frame_width, frame_height = (480,480)
    frame = cv2.resize(frame,(frame_width, frame_height))
    #frame =cv2.flip(frame,-1)
    center_frame = (frame_width//2, frame_height//2)
    blurred = cv2.GaussianBlur(frame,(3,3),0)
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #Calculate FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame,"FPS:{}".format(int(fps)),(15,15),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)#Displays fps

    target_track_radius = 50
    cv2.circle(frame, (frame_width//2, frame_height//2), target_track_radius, (255,0,0), 1)
    cv2.line(frame,(int(frame_width/2 ),0),(int(frame_width/2 ),int(frame_height)),(255,0,0),1)
    cv2.line(frame,(0,int(frame_height/2 )),(int(frame_width),int(frame_height/2 )),(255,0,0),1)
    

    min_h = cv2.getTrackbarPos("min - H", "ColorTrackbars")
    min_s = cv2.getTrackbarPos("min - S", "ColorTrackbars")
    min_v = cv2.getTrackbarPos("min - V", "ColorTrackbars")
    max_h = cv2.getTrackbarPos("max - H", "ColorTrackbars")
    max_s = cv2.getTrackbarPos("max - S", "ColorTrackbars")
    max_v = cv2.getTrackbarPos("max - V", "ColorTrackbars")
    min_color = np.array([min_h, min_s, min_v])
    max_color = np.array([max_h, max_s, max_v])
    mask_color = cv2.inRange(hsv_frame, min_color, max_color)

    contours, hierarchy = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #SIMPLE
    #contours = max(contours, key = cv2.contourArea)
    
    contours = sorted(contours, key = cv2.contourArea)
    contours = contours[-1:] # Take the object with the largest area

    for contour in contours:
        if cv2.contourArea(contour) >= 500: # If area is big enough, find its center etc.
            contour = cv2.approxPolyDP(contour, 10, closed=True)
            cv2.drawContours(frame, contour, -1, (255,255,0), 5, lineType = cv2.FILLED)
            print("len(contour): ",len(contour))
            cv2.polylines(frame, [contour], True, (0,255,255), 2)
            """
            x,y,w,h = cv2.boundingRect(contour)
            x,y,w,h = (int(x),int(y),int(w),int(h))
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.line(frame, (x+(w//2),frame_height),(x+(w//2),frame_width ),(0,255,0),3)
            """

            moment = cv2.moments(contour) # To find the center of the contour, we use cv2.moment
            (x_contour, y_contour) = (moment['m10'] / (moment['m00'] + 1e-5), moment['m01'] / (moment['m00'] + 1e-5)) # calculate center of the contour
            center_contour = (int(x_contour), int(y_contour))
            cv2.circle(frame, center_contour, 5, (255,0,255),2)

            cv2.line(frame, center_frame, center_contour,(0,0,255),1)
            atan = math.atan2(center_frame[1]-center_contour[1], center_frame[0]-center_contour[0])
            angle = math.degrees(atan)
            angle = int(90-angle)
            cv2.putText(frame, "{}*".format(angle), center_frame , cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    cv2.imshow("realTimeCamera",frame)    
    cv2.imshow("mask_color",mask_color)
    
    #cv2.imshow("Blurred",blurred)
    key=cv2.waitKey(1)
    if key==27:
        break
cv2.destroyAllWindows()
cam.release()