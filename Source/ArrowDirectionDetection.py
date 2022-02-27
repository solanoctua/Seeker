import cv2, time, math
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

cam = cv2.VideoCapture(0)
color = (255,255,255)
def nothing(x):
    pass
cv2.namedWindow("CornerTrackbars")
cv2.createTrackbar("qualityLevel", "CornerTrackbars", 1, 49, nothing)
cv2.createTrackbar("maxCorners", "CornerTrackbars", 5, 20, nothing)
cv2.createTrackbar("minDistance", "CornerTrackbars", 10, 15, nothing)
cv2.createTrackbar("blockSize", "CornerTrackbars", 3, 15, nothing)
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
else: 
    ret = False
while ret :
    ret,frame = cam.read()
    frame = cv2.imread("Seeker/TargetImages/arrow1.png")
    frame_width, frame_height = (480,480)
    frame = cv2.resize(frame,(frame_width, frame_height))
    #frame =cv2.flip(frame,-1)
    center_frame = (frame_width//2, frame_height//2)
    blurred = cv2.GaussianBlur(frame,(3,3),0)
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
    target_lock_radius = 100
    #cv2.circle(frame, (frame_width//2, frame_height//2), target_lock_radius, (255,0,0), 1)
    cv2.line(frame,(int(frame_width/2 +target_lock_radius),0),(int(frame_width/2 +target_lock_radius),int(frame_height)),(255,0,0),1)
    cv2.line(frame,(0,int(frame_height/2 + target_lock_radius)),(int(frame_width),int(frame_height/2 + target_lock_radius)),(255,0,0),1)
    cv2.line(frame,(int(frame_width/2 -target_lock_radius),0),(int(frame_width/2-target_lock_radius),int(frame_height)),(255,0,0),1)
    cv2.line(frame,(0,int(frame_height/2 - target_lock_radius)),(int(frame_width),int(frame_height/2 - target_lock_radius)),(255,0,0),1)
    # convert to HSV colorspace 
    
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
    #white = cv2.bitwise_and(frame, frame, mask = mask_white)

    contours, hierarchy = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) #SIMPLE
    #contours = max(contours, key = cv2.contourArea)
    
    contours = sorted(contours, key = cv2.contourArea)
    contours = contours[-1:] # Take the object with the largest area
    
    """
    contours, hierarchy = cv2.findContours(hsv_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x:cv2.contourArea(x), reverse=True)
    """
            
    for contour in contours:
        if cv2.contourArea(contour) >= 500: # If area is big enough, find its center etc.
            contour = cv2.approxPolyDP(contour, 10, closed=True)
            #print(len(contour))
            if 7 < len(contour) < 20 :
                #print("sign")
                pass
                
            cv2.drawContours(frame, contour, -1, (255,255,0), 15, lineType = cv2.FILLED)
            
            moment = cv2.moments(contour) # To find the center of the contour, we use cv2.moment
            (x_contour, y_contour) = (moment['m10'] / (moment['m00'] + 1e-5), moment['m01'] / (moment['m00'] + 1e-5)) # calculate center of the contour
            center_contour = (int(x_contour), int(y_contour))
            cv2.circle(frame, center_contour, 5, (255,0,255),2)
                    
            (x_contour_circle, y_contour_circle), radius = cv2.minEnclosingCircle(contour)
            radius = int(radius)
            center_contour_circle = (int(x_contour_circle), int(y_contour_circle))
            cv2.circle(frame, center_contour_circle, radius, (255,0,255),2)
            #Find corners
            mask_arrow = np.ones(frame.shape[:2], dtype="uint8") * 255
            
            #cv2.drawContours(mask_arrow, contour, -1, (0,0,0), 15, lineType = cv2.FILLED)
            ConvexHullPoints = cv2.convexHull(contour)
            #cv2.fillPoly(mask_arrow, ConvexHullPoints, color=(0,0,0))
            cv2.polylines(mask_arrow, [ConvexHullPoints], True, (0,255,255), 2)
            
            set_qualityLevel = cv2.getTrackbarPos("qualityLevel", "CornerTrackbars")
            set_maxCorners = cv2.getTrackbarPos("maxCorners", "CornerTrackbars")
            set_minDistance = cv2.getTrackbarPos("minDistance", "CornerTrackbars")
            set_blockSize = cv2.getTrackbarPos("blockSize", "CornerTrackbars")
            
            set_qualityLevel /= 100
            corners = cv2.goodFeaturesToTrack(mask_arrow, maxCorners = set_maxCorners, qualityLevel = set_qualityLevel ,minDistance = set_minDistance, blockSize = set_blockSize, useHarrisDetector = 1, k = 0.04)
            
            if corners is not None:
                if len(corners) >= 5:
                    corners = np.int0(corners)
                    mean_x = 0
                    mean_y = 0
                    for i in corners:
                        x,y = i.ravel()
                        cv2.circle(frame,(x,y),5,(0,0,255),-1)
                        mean_x += x
                        mean_y += y
                        
                    x = mean_x // 5
                    y = mean_y // 5
                    
                    center = (int(x), int(y))
                    cv2.circle(frame,(int(center[0]), int(center[1])), 5,(0,255,0), -1)
                    cv2.circle(mask_arrow,(int(center[0]), int(center[1])), 5,(0,255,0), -1)
                    cv2.putText(frame, "corners midpoint({},{})".format(center[0],center[1]),(int(x)+15, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                    #print("mid point", center)        
                    #Drawing lines
                    cv2.line(frame, center_contour, center,(255,0,255),1)
                    cv2.line(frame, center_frame, center_contour,(0,0,255),1)

                    #Angles
                    atan = math.atan2(center[1]-center_contour[1], center[0]-center_contour[0])
                    angle = math.degrees(atan)
                    angle = int(angle)
                    cv2.putText(frame, "{}*".format(angle), center_contour , cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                    #print ("angle = ", angle)
                    kernel = np.ones((5,5), np.uint8)
                    mask_color = cv2.erode(mask_color, kernel, iterations=1)
                    mask_color = cv2.dilate(mask_color, kernel, iterations=1)
                    ret,mask_color = cv2.threshold(np.array(mask_color), 125, 255, cv2.THRESH_BINARY_INV)
                    #print("text: ",pytesseract.image_to_string(mask_color, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')) 
                    print("text: ",pytesseract.image_to_string(mask_color, config='digits'))
                    
                    
                    
            
            target_color = cv2.bitwise_and(frame, frame, mask = mask_color)
            #draw a transparent rectangle to the zone, where our targeted object's center point lies.
            if(center_contour[0] <= int(frame_width/2 -target_lock_radius) and center_contour[1] <= int(frame_height/2 - target_lock_radius)):
                #print("ZONE 1")
                #To draw a rectangle, you need top-left corner and bottom-right corner of rectangle. 
                cv2.rectangle(blank,(0,0),(int(frame_width/2 -target_lock_radius),int(frame_height/2 - target_lock_radius)),(0,255,0),cv2.FILLED)
                                            
            elif(center_contour[0] <= int(frame_width/2 -target_lock_radius) and center_contour[1] >= int(frame_height/2 - target_lock_radius) and center_contour[1] <= int(frame_height/2 + target_lock_radius)):
                #print("ZONE 2")
                cv2.rectangle(blank,(0,int(frame_height/2 - target_lock_radius)),(int(frame_width/2 -target_lock_radius),int(frame_height/2 + target_lock_radius)),(0,255,0),cv2.FILLED)
            elif(center_contour[0] <= int(frame_width/2 -target_lock_radius) and center_contour[1] >= int(frame_height/2 + target_lock_radius)):
                #print("ZONE 3")
                cv2.rectangle(blank,(0,int(frame_height/2 + target_lock_radius)),(int(frame_width/2 -target_lock_radius),int(frame_height)),(0,255,0),cv2.FILLED)
            elif(center_contour[0] >= int(frame_width/2 -target_lock_radius) and center_contour[0] <= int(frame_width/2 +target_lock_radius) and center_contour[1] <= int(frame_height/2 - target_lock_radius)  ):
                #print("ZONE 4")
                cv2.rectangle(blank,(int(frame_width/2 -target_lock_radius),0),(int(frame_width/2 +target_lock_radius),int(frame_height/2 - target_lock_radius)),(0,255,0),cv2.FILLED)
            elif(center_contour[0] >= int(frame_width/2 -target_lock_radius) and center_contour[0] <= int(frame_width/2 +target_lock_radius) and center_contour[1] >= int(frame_height/2 - target_lock_radius) and center_contour[1] <= int(frame_height/2 + target_lock_radius)):
                #print("ZONE 5")
                cv2.rectangle(blank,(int(frame_width/2 -target_lock_radius),int(frame_height/2 - target_lock_radius)),(int(frame_width/2 + target_lock_radius),int(frame_height/2 + target_lock_radius)),(0,255,0),cv2.FILLED)
            elif(center_contour[0] >= int(frame_width/2 -target_lock_radius) and center_contour[0] <= int(frame_width/2 +target_lock_radius) and center_contour[1] >= int(frame_height/2 + target_lock_radius)  ):
                #print("ZONE 6")
                cv2.rectangle(blank,(int(frame_width/2 -target_lock_radius),int(frame_height/2 + target_lock_radius)),(int(frame_width/2 +target_lock_radius),int(frame_height)),(0,255,0),cv2.FILLED)
            elif(center_contour[0] >= int(frame_width/2 +target_lock_radius) and center_contour[1] <= int(frame_height/2 - target_lock_radius)):
                #print("ZONE 7")
                cv2.rectangle(blank,(int(frame_width/2 +target_lock_radius),0),(int(frame_width),int(frame_height/2 - target_lock_radius)),(0,255,0),cv2.FILLED)
            elif(center_contour[0] >= int(frame_width/2 +target_lock_radius) and center_contour[1] >= int(frame_height/2 - target_lock_radius) and center_contour[1] <= int(frame_height/2 + target_lock_radius)):
                #print("ZONE 8")
                cv2.rectangle(blank,(int(frame_width/2 +target_lock_radius),int(frame_height/2 - target_lock_radius)),(int(frame_width),int(frame_height/2 + target_lock_radius)),(0,255,0),cv2.FILLED)
            elif(center_contour[0] >= int(frame_width/2 +target_lock_radius) and center_contour[1] >= int(frame_height/2 + target_lock_radius)):
                #print("ZONE 9")
                cv2.rectangle(blank,(int(frame_width/2 +target_lock_radius),int(frame_height/2 + target_lock_radius)),(int(frame_width),int(frame_height)),(0,255,0),cv2.FILLED)
            else:
                pass        
            alpha = 0.4
            beta = (1.0 - alpha)
            cv2.addWeighted(blank, alpha, frame, beta, 0.0,frame) # to make rectangle transparent
            cv2.imshow("mask_arrow",mask_arrow)
    
    cv2.imshow("realTimeCamera",frame)    
    cv2.imshow("mask_color",mask_color)
    
    #cv2.imshow("Blurred",blurred)
    key=cv2.waitKey(1)
    if key==27:
        break
cv2.destroyAllWindows()
cam.release()




    
