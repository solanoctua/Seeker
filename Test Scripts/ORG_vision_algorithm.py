import cv2, time, math
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

frame_width, frame_height = (640,640)  #1280,720
target_lock_radius = 275
kernel = np.ones((3, 3), 'uint8') # for morphological operations
# Parameters for black detection in HSV colorspace
min_h, min_s, min_v = 0,0,0
max_h, max_s, max_v = 179,255,100
min_color = np.array([min_h, min_s, min_v])
max_color = np.array([max_h, max_s, max_v])

color = (255,255,255)

def distance_between_points(point1, point2):
    return np.sqrt(np.power(point1[0]-point2[0],2) + np.power(point1[1]-point2[1],2))

def findPointer(points): # For finding arrow direction and angle
    temp = []
    for point1 in points:
        for point2 in points:
            distance = int(distance_between_points(point1, point2))
            temp.append((distance,point1,point2))
            
    #print("sorted: ",sorted(temp,key=lambda x: (x[0]),reverse=True))
    sortedbydistance = sorted(temp,key=lambda x: (x[0]),reverse=True)
    special_points = []
    for x in sortedbydistance[0:3]:
        if x[1] not in special_points:
            special_points.append(x[1])
        if x[2] not in special_points:
            special_points.append(x[2])
    #print("specials: ",special_points)

    mean = ((special_points[0][0]+special_points[1][0]+special_points[2][0])//3,(special_points[0][1]+special_points[1][1]+special_points[2][1])//3)
    max = 0
    for point in special_points:
        d = distance_between_points(point,mean)
        if d > max :
            max = d
            pointer = point
    special_points.remove(pointer)
    special_points.insert(0,pointer)
    return special_points
def calculateArrowDirection(contour):

    # To find Arrow direction, first extract contour points
    points = contour.ravel()
    temp = []
    Points = []
    for i in range (0,len(points)):
        temp.append(points[i])
        if i%2 == 1:
            Points.append(temp)
            temp = []
        else: 
            continue

    specials = findPointer(Points)
    middle = ((specials[1][0] + specials[2][0])//2, (specials[1][1] + specials[2][1])//2)
    pointer = (specials[0][0], specials[0][1])
    
    # ONLY FOR VISUAL PURPOSES
    cv2.circle(frame, (specials[0][0],specials[0][1]), 4, (0,0,255),-1)
    cv2.circle(frame, (specials[1][0],specials[1][1]), 4, (0,255,255),-1)
    cv2.circle(frame, (specials[2][0],specials[2][1]), 4, (0,255,255),-1)

    # Drawing lines for angle calculation (for visual purposes only)
    cv2.line(frame, middle, pointer,(255,0,255),1)
    #cv2.line(frame, center_contour, pointer,(255,0,255),1)
    cv2.line(frame, center_frame, center_contour,(0,0,255),1)
    # Find angle of the arrow
    atan = math.atan2(middle[0] - pointer[0], middle[1] - pointer[1])
    angle_arrow = math.degrees(atan)
    angle_arrow = int(angle_arrow)
    if angle_arrow > 0:
        if angle_arrow > 90:
            angle_arrow = 270-(angle_arrow-90)
        else:
            angle_arrow = 360 - angle_arrow
    else:
        angle_arrow *= -1

    return angle_arrow

def searchForText(contour, tolerance):
    ROI = cv2.minAreaRect(contour)
    ROI = cv2.boxPoints(ROI)
    ROI = np.int0(ROI) 
    
    min_height = frame_height
    max_height = 0
    min_width = frame_width
    max_width = 0
    for i in range(0,4):
        if ROI[i][0] < min_height:
            min_height = ROI[i][0]
        if ROI[i][0] > max_height:
            max_height = ROI[i][0]
        if ROI[i][1] < min_width:
            min_width = ROI[i][1]
        if ROI[i][1] > max_width:
            max_width = ROI[i][1]

    #print("{}:{} , {}:{}".format(min_height, max_height, min_width, max_width))
    if min_height - tolerance >= 0:
        min_height -= tolerance
    if max_height + tolerance <= frame_height:
        max_height += tolerance
    if min_width - tolerance >= 0:
        min_width -= tolerance
    if max_width + tolerance <= frame_width:
        max_width += tolerance
    
    # Cropping text area as an input to OCR
    text_area = mask_color_inv[ min_width : max_width , min_height : max_height]
    # Read the text https://muthu.co/all-tesseract-ocr-options/
    #print("text: ",pytesseract.image_to_string(text)) #, config='digits'
    #text = pytesseract.image_to_string(text_area, lang='eng',config='--psm 6') #--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789
    try:
        text = pytesseract.image_to_string(text_area, lang='eng',config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789HXLT')
    except:
        #print("cannot read the text")
        text = ""
    cv2.drawContours(frame,[ROI],0,(0,0,255),2)
    text = text.replace(" ", "")
    #print("text: ",text)
    #print("len(text[]): ",len(text[:-1]))
    cv2.imshow("text", text_area)
    return text

def calculateAngleOfTarget(center_contour):
    
    # ONLY FOR VISUAL PURPOSES
    cv2.circle(frame, center_contour, 10, (0,255,0),-1)

    # Angle of the line connecting center of contour to the center of the frame
    cv2.line(frame, center_frame, center_contour,(0,0,255),1) # for visual
    atan = math.atan2(center_frame[1] - center_contour[1], center_frame[0] - center_contour[0])
    angle_target = math.degrees(atan)
    angle_target = int(angle_target)
    if angle_target > 0:
        if angle_target > 90:
            angle_target -= 90
        else:
            angle_target += 270
    else:
        angle_target += 270

    cv2.putText(frame, "{}*".format(angle_target), center_frame , cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)# for visual
    return angle_target

cam = cv2.VideoCapture(0)
mission = "X"
prev_frame_time = 0
new_frame_time = 0
if cam.isOpened():
    ret,frame = cam.read()
    fps = int(cam.get(cv2.CAP_PROP_FPS))
    #print("FPS: ",fps)
    output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), 2.0, (frame_width, frame_height)) # 'M','J','P','G' #https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html
else: 
    ret = False
while ret :
    ret,frame = cam.read()
    #frame = cv2.imread("Seeker/TargetImages/arrow1.png")
    frame = cv2.resize(frame,(frame_width, frame_height ))
    #frame =cv2.flip(frame,-1)
    #Calculate FPS
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    
    # Find center of the frame for locking targets
    center_frame = (frame_width//2,frame_height//2)
    # convert BGR colorspace to HSV colorspace 
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # ONLY FOR VISUAL PURPOSES
    blank = np.zeros(frame.shape, np.uint8)

    # Detect colors only in range that we previously specified
    mask_color = cv2.inRange(hsv_frame, min_color, max_color)
    _, mask_color_inv = cv2.threshold(mask_color, 127, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #SIMPLE-NONE
    contours = sorted(contours, key = cv2.contourArea)
    try:
        target_contours = contours[-3:] # Take the object with the largest area
    except:
        target_contours = contours[-1:]
    for contour in target_contours:
        contour = cv2.approxPolyDP(contour, 10, closed=True)
        
        if cv2.contourArea(contour) >= 300: # If area is big enough, find its center etc.
            cv2.drawContours(frame, contour, -1, (255,0,0), 15, lineType = cv2.FILLED)# for visual
            #print("len(contour): ",len(contour))
            if 7 < len(contour) < 32  :
                #print("sign")
                # Find center of the contour
                moment = cv2.moments(contour) # To find the center of the contour, we use cv2.moment
                (x_contour, y_contour) = (moment['m10'] / (moment['m00'] + 1e-5), moment['m01'] / (moment['m00'] + 1e-5)) # calculate center of the contour
                center_contour = (int(x_contour), int(y_contour))

                # Calculate angle of the target wrt QUAD frame
                angle_target = calculateAngleOfTarget(center_contour)
                # Go to the center of the sign symbol
                if distance_between_points(center_contour, center_frame) >= target_lock_radius:
                    print("Sign is not in lock radius!")
                    #condition_yaw(angle_target)
                    print("condition_yaw({})".format(angle_target))
                    velocity_x = 0.1 # in meters
                    velocity_y = 0  
                    velocity_z = 0
                    duration = 1
                    #send_body_ned_velocity(velocity_x, velocity_y, velocity_z, duration)
                    print("send_body_ned_velocity({}, {}, {}, {})".format(velocity_x, velocity_y, velocity_z, duration))

                if distance_between_points(center_contour, center_frame) < target_lock_radius:    
                    try:
                        mission = searchForText(contour, tolerance = -70)
                    except:
                        print("issue for setting mission")

                    #print("mission: ", mission)
                
            if (mission == "X" and len(contour) == 7 ) :
                print("Executing follow arrows mission")
                # Find center of the contour
                moment = cv2.moments(contour) # To find the center of the contour, we use cv2.moment
                (x_contour, y_contour) = (moment['m10'] / (moment['m00'] + 1e-5), moment['m01'] / (moment['m00'] + 1e-5)) # calculate center of the contour
                center_contour = (int(x_contour), int(y_contour))
                # Calculate angle of the target wrt QUAD frame
                angle_target = calculateAngleOfTarget(center_contour)
                # Go to the center of the sign symbol
                """
                    FORWARD: Yaw 0 absolute (North)
                    BACKWARD: Yaw 180 absolute (South)
                    LEFT: Yaw 270 absolute (West)
                    RIGHT: Yaw 90 absolute (East)
                """
                
                if distance_between_points(center_contour, center_frame) >= target_lock_radius:
                    print("Arrow is not in lock radius!")
                    #condition_yaw(angle_target)
                    print("condition_yaw({})".format(angle_target))
                    velocity_x = 0.1 # in meters
                    velocity_y = 0  
                    velocity_z = 0
                    duration = 1
                    #send_body_ned_velocity(velocity_x, velocity_y, velocity_z, duration)
                    print("send_body_ned_velocity({}, {}, {}, {})".format(velocity_x, velocity_y, velocity_z, duration))
                    #time.sleep(1)
            
                # If arrow inside the locking_circle, then locking_circle becomes green
                if distance_between_points(center_contour, center_frame) < target_lock_radius:
                    # ONLY FOR VISUAL PURPOSES
                    cv2.circle(blank, center_frame, target_lock_radius, (0,255,0), cv2.FILLED)
                    alpha = 0.4
                    beta = (1.0 - alpha)
                    cv2.addWeighted(blank, alpha, frame, beta, 0.0, frame) # to make rectangle transparent

                    velocity_x = 0 # in meters
                    velocity_y = 0  
                    velocity_z = 0
                    duration = 1 
                    #send_body_ned_velocity(velocity_x, velocity_y, velocity_z, duration)
                    print("send_body_ned_velocity({}, {}, {}, {})".format(velocity_x, velocity_y, velocity_z, duration))
                    #time.sleep(1)
                    # Find angle of the arrow
                    angle_arrow = calculateArrowDirection(contour)
                    # ONLY FOR VISUAL PURPOSES
                    color = (0,0,255)
                    if angle_arrow > 90 and angle_arrow < 270:
                        cv2.putText(frame, "BACKWARD", (frame_width - 100, 35) , cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    if angle_arrow > 180 and angle_arrow < 360:
                        cv2.putText(frame, "LEFT", (frame_width - 100, 55) , cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    if angle_arrow >0 and angle_arrow < 180:
                        cv2.putText(frame, "RIGHT", (frame_width - 100, 55) , cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    if  angle_arrow < 90 or angle_arrow > 270:
                        cv2.putText(frame, "FORWARD", (frame_width - 100, 35) , cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    cv2.putText(frame, "Arrow Direction: {}*".format(angle_arrow), (frame_width - 200, 15) , cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    
                    # Dilate to connect text characters 
                    mask_color = cv2.dilate(mask_color, kernel, iterations =4) 
                    # Find all text as a one contour
                    contours, hierarchy = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #SIMPLE-NONE
                    contours = sorted(contours, key = cv2.contourArea)
                    target_contours = contours[-2:-1] # -2:-1Take the object with the second largest area
                    for contour in target_contours:
                        if cv2.contourArea(contour) >= 300: # If area is big enough, find its center etc.

                            # Adjust the angle of the frame wrt arrows angle
                            #condition_yaw(angle)
                            print("condition_yaw({})".format(angle_arrow))

                            # Find smallest rectangle that encloses the text
                            text = searchForText(contour, tolerance = 10)
                            try:
                                text = int(text[:-1])
                            except:
                                print("cannot convert the text to int")
                            if text:
                                #print("found",text) 
                                cv2.putText(frame, "arrow {}".format(text), (10, frame_height-35) , cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)
                                
                                distance = text/100 # in meters
                                print("Go {} meters in yaw direction {}".format(distance, angle_arrow))
                                # Read the distance value and go to the specified direction with given distance
                                velocity_x = 0.2 # in meters
                                velocity_y = 0  
                                velocity_z = 0
                                duration = distance/velocity_x 
                                print("forward at {} m/s for {} sec.".format(velocity_x, duration))
                                print("send_body_ned_velocity({}, {}, {}, {})".format(velocity_x, velocity_y, velocity_z, duration))
                                #send_body_ned_velocity(velocity_x, velocity_y, velocity_z, duration)
                                
                            else:
                                print("cannot read distance")
                            
            if (mission == "L" and len(contour) <= 6) :
                print("Executing follow line mission")
                angle_target = calculateAngleOfTarget(center_contour)
                print("condition_yaw({})".format(angle_target))

                
                # Go to the center of the T symbol
                if distance_between_points(center_contour, center_frame) >= target_lock_radius:
                    velocity_x = 0.1 # in meters
                    velocity_y = 0  
                    velocity_z = 0
                    duration = 1
                    #send_body_ned_velocity(velocity_x, velocity_y, velocity_z, duration)
                    print("send_body_ned_velocity({}, {}, {}, {})".format(velocity_x, velocity_y, velocity_z, duration))
                    #time.sleep(1)
               
                if distance_between_points(center_contour, center_frame) < target_lock_radius:
                    # ONLY FOR VISUAL PURPOSES
                    cv2.circle(blank, center_frame, target_lock_radius, (0,255,0), cv2.FILLED)
                    alpha = 0.4
                    beta = (1.0 - alpha)
                    cv2.addWeighted(blank, alpha, frame, beta, 0.0, frame) # to make rectangle transparent

            if mission == "T":
                print("Executing land mission")
                angle_target = calculateAngleOfTarget(center_contour)
                print("condition_yaw({})".format(angle_target))

                velocity_x = 0.1 # in meters
                velocity_y = 0  
                velocity_z = 0
                # Go to the center of the T symbol
                if distance_between_points(center_contour, center_frame) >= target_lock_radius:
                    duration = 1
                    #send_body_ned_velocity(velocity_x, velocity_y, velocity_z, duration)
                    print("send_body_ned_velocity({}, {}, {}, {})".format(velocity_x, velocity_y, velocity_z, duration))
                    #time.sleep(1)
               
                if distance_between_points(center_contour, center_frame) < target_lock_radius:
                    # ONLY FOR VISUAL PURPOSES
                    cv2.circle(blank, center_frame, target_lock_radius, (0,255,0), cv2.FILLED)
                    alpha = 0.4
                    beta = (1.0 - alpha)
                    cv2.addWeighted(blank, alpha, frame, beta, 0.0, frame) # to make rectangle transparent
                    """
                    vehicle.mode = VehicleMode("LAND")
                    disarm(wait=True, timeout=None)
                    vehicle.close()
                    """
            

    cv2.putText(frame,"FPS:{}".format(int(fps)),(15,15),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1,cv2.LINE_AA)# Displays fps
    cv2.putText(frame, "mission: "+ mission, (10, frame_height-55) , cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)
    cv2.circle(frame, (frame_width//2, frame_height//2), target_lock_radius, (0,255,0), 1) # target lock circle
    cv2.line(frame,(int(frame_width/2),0),(int(frame_width/2),int(frame_height)),(0,255,0),1) # vertical line
    cv2.line(frame,(0,int(frame_height/2)),(frame_width,int(frame_height/2)),(0,255,0),1) # horizontal line
    output.write(frame) 
    cv2.imshow("realTimeCamera", frame)    
    #cv2.imshow("mask_color", mask_color)
    key=cv2.waitKey(1)
    if key==27:
        break
cv2.destroyAllWindows()
output.release()
cam.release()

