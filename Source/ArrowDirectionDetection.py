import cv2, time, math
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

cam = cv2.VideoCapture(0)
color = (255,255,255)
def distance_between_points(point1, point2):
    return np.sqrt(np.power(point1[0]-point2[0],2) + np.power(point1[1]-point2[1],2))
def nothing(x):
    pass

"""
cv2.namedWindow("ColorTrackbars")
cv2.createTrackbar("min - H", "ColorTrackbars", 0, 179, nothing)
cv2.createTrackbar("min - S", "ColorTrackbars", 0, 255, nothing)
cv2.createTrackbar("min - V", "ColorTrackbars", 240, 255, nothing)
cv2.createTrackbar("max - H", "ColorTrackbars", 179, 179, nothing)
cv2.createTrackbar("max - S", "ColorTrackbars", 255, 255, nothing)
cv2.createTrackbar("max - V", "ColorTrackbars", 255, 255, nothing)
"""
cv2.namedWindow("ColorTrackbars")
cv2.createTrackbar("min - H", "ColorTrackbars", 0, 179, nothing)
cv2.createTrackbar("min - S", "ColorTrackbars", 0, 255, nothing)
cv2.createTrackbar("min - V", "ColorTrackbars", 0, 255, nothing)
cv2.createTrackbar("max - H", "ColorTrackbars", 179, 179, nothing)
cv2.createTrackbar("max - S", "ColorTrackbars", 255, 255, nothing)
cv2.createTrackbar("max - V", "ColorTrackbars", 100, 255, nothing)
prev_frame_time = 0
new_frame_time = 0
if cam.isOpened():
    ret,frame = cam.read()
    frame_width, frame_height = (640,640)
    output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width, frame_height)) #https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html
else: 
    ret = False
while ret :
    ret,frame = cam.read()
    frame = cv2.imread("Seeker/TargetImages/arrow1.png")
    frame = cv2.resize(frame,(frame_width, frame_height ))
    
    #frame =cv2.flip(frame,-1)
    center_frame = (frame_width//2,frame_height//2)
    blurred = cv2.GaussianBlur(frame,(3,3),0)
    # convert to HSV colorspace 
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blank = np.zeros(frame.shape, np.uint8)
    blank2 = np.zeros(frame.shape, np.uint8)
    #Calculate FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    cv2.putText(frame,"FPS:{}".format(int(fps)),(15,15),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)#Displays fps
    # lines for left,right,up,down boundaries
    #cv2.circle(frame , center_frame, 15,(0,255,0), 1)
    target_lock_radius = 75
    cv2.circle(frame, (frame_width//2, frame_height//2), target_lock_radius, (0,255,0), 1)
    cv2.line(frame,(int(frame_width/2),0),(int(frame_width/2),int(frame_height)),(0,255,0),1) # vertical line
    cv2.line(frame,(0,int(frame_height/2)),(frame_width,int(frame_height/2)),(0,255,0),1) # horizontal line
    
    #H,S,V = cv2.split(hsv_frame)
    min_h = cv2.getTrackbarPos("min - H", "ColorTrackbars")
    min_s = cv2.getTrackbarPos("min - S", "ColorTrackbars")
    min_v = cv2.getTrackbarPos("min - V", "ColorTrackbars")
    max_h = cv2.getTrackbarPos("max - H", "ColorTrackbars")
    max_s = cv2.getTrackbarPos("max - S", "ColorTrackbars")
    max_v = cv2.getTrackbarPos("max - V", "ColorTrackbars")
    min_color = np.array([min_h, min_s, min_v])
    max_color = np.array([max_h, max_s, max_v])
    mask_color = cv2.inRange(hsv_frame, min_color, max_color)

    contours, hierarchy = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #SIMPLE-NONE

    #contours = max(contours, key = cv2.contourArea)
    contours = sorted(contours, key = cv2.contourArea)
    target_contours = contours[-1:] # Take the object with the largest area
    """
    contours, hierarchy = cv2.findContours(hsv_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
    """    
    for contour in target_contours:
        if cv2.contourArea(contour) >= 500: # If area is big enough, find its center etc.
            contour = cv2.approxPolyDP(contour, 10, closed=True)
            
            #print(len(contour))
            if 7 < len(contour) < 20 :
                print("sign")
                pass
                
            cv2.drawContours(frame, contour, -1, (255,0,0), 15, lineType = cv2.FILLED)
            # Find pointer(sharp point of the arrow) 
            points = contour.ravel()
            pointer = (int(points[0]), int(points[1]))
            cv2.circle(frame, pointer, 5, (0,0,255),-1)
            # Find center of the contour
            moment = cv2.moments(contour) # To find the center of the contour, we use cv2.moment
            (x_contour, y_contour) = (moment['m10'] / (moment['m00'] + 1e-5), moment['m01'] / (moment['m00'] + 1e-5)) # calculate center of the contour
            center_contour = (int(x_contour), int(y_contour))
            cv2.circle(frame, center_contour, 10, (0,255,0),-1)
            # Find corners
            mask_arrow = np.ones(frame.shape[:2], dtype="uint8") * 255
            #cv2.drawContours(mask_arrow, contour, -1, (0,0,0), 15, lineType = cv2.FILLED)
            cv2.polylines(mask_arrow, [contour], True, (0,255,255), 2)#ConvexHullPoints
            # Drawing lines for angle calculation (for visual purposes only)
            cv2.line(frame, center_contour, pointer,(255,0,255),1)
            cv2.line(frame, center_frame, center_contour,(0,0,255),1)
            # Find angle of the arrow
            atan = math.atan2(pointer[1]-center_contour[1], pointer[0]-center_contour[0])
            angle_arrow = math.degrees(atan)
            angle_arrow = int(90+angle_arrow)
            # Angle of the line connecting center of contour to the center of the frame
            atan = math.atan2(center_frame[1]-center_contour[1], center_frame[0]-center_contour[0])
            angle_target = math.degrees(atan)
            angle_target = int(360- (90-angle_target) ) #-90
            color = (0,0,255)
            if 0 <= np.abs(angle_arrow) and np.abs(angle_arrow) <= 90:
                cv2.putText(frame, "FORWARD", (frame_width - 100, 35) , cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if 90 < np.abs(angle_arrow):
                cv2.putText(frame, "BACKWARD", (frame_width - 100, 35) , cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if angle_arrow < 0:
                cv2.putText(frame, "LEFT", (frame_width - 100, 55) , cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            if angle_arrow > 0:
                cv2.putText(frame, "RIGHT", (frame_width - 100, 55) , cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            cv2.putText(frame, "{}*".format(angle_arrow), (frame_width - 100, 15) , cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.putText(frame, "{}*".format(angle_target-angle_arrow), center_frame , cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            # If arrow inside the locking_circle, then locking_circle becomes green
            if distance_between_points(center_contour, center_frame) < target_lock_radius:
                cv2.circle(blank, center_frame, target_lock_radius,(0,255,0), cv2.FILLED)
                alpha = 0.4
                beta = (1.0 - alpha)
                cv2.addWeighted(blank, alpha, frame, beta, 0.0,frame) # to make rectangle transparent

            kernel = np.ones((5,5), np.uint8)
            mask_color = cv2.erode(mask_color, kernel, iterations=1)
            mask_color = cv2.dilate(mask_color, kernel, iterations=1)
            ret, mask_color = cv2.threshold(np.array(mask_color), 125, 255, cv2.THRESH_BINARY_INV)
            #print("text: ",pytesseract.image_to_string(mask_color, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')) 
            #print("text: ",pytesseract.image_to_string(mask_color, config='digits'))
    output.write(frame)    
    cv2.imshow("mask_arrow",mask_arrow)
    cv2.imshow("realTimeCamera",frame)    
    cv2.imshow("mask_color",mask_color)
    
    #cv2.imshow("Blurred",blurred)
    key=cv2.waitKey(1)
    if key==27:
        break
cv2.destroyAllWindows()
output.release()
cam.release()





    
