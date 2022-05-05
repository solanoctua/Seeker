import cv2 
import numpy as np
import math
import time

img_object = cv2.imread("C:/Users/asus/Desktop/Samurai-Seeker/VITA.png", cv2.IMREAD_GRAYSCALE)

if img_object is None:
    print('Could not open or find the images!')
    exit(0)

# Initiate SIFT detector
detector = cv2.xfeatures2d.SIFT_create()
keypoints_object, descriptors_object = detector.detectAndCompute(img_object, None)
output = cv2.drawKeypoints(img_object, keypoints_object, None)
"""
#Step 1: Detect the keypoints using SURF Detector, compute the descriptors
minHessian = 300
detector = cv2.xfeatures2d_SURF.create(hessianThreshold=minHessian)
keypoints_object, descriptors_object = detector.detectAndCompute(img_object, None)
output = cv2.drawKeypoints(img_object, keypoints_object, None)
"""

#-- Step 2: Matching descriptor vectors with a FLANN based matcher

matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

#Capturing Real Time Video
cap = cv2.VideoCapture(0)
prev_frame_time = 0
new_frame_time = 0
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
    
    cv2.imshow("Searching object", output)
    ret , frame = cap.read()
    blank = np.zeros(frame.shape, np.uint8) #black image of same dimensions as frame, for drawing the rectangles in the future
    frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   # float `width`
    frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    #print(frame_width,frame_height)

    # Calculating the fps 
    new_frame_time = time.time() 
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    cv2.putText(frame,"fps: "+str(int(fps)),(15,15), cv2.FONT_HERSHEY_SIMPLEX, .6,(0,0,255),1,cv2.LINE_AA) # displays fps

    blur = cv2.GaussianBlur(frame,(3,3),0) # blured (filtered) image with a 5x5 gaussian kernel to remove the noise
    grayFrame = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)   #GRAYSCALE
    keypoints_scene, descriptors_scene = detector.detectAndCompute(grayFrame, None)
    """
    show_keypoints_scene = cv2.drawKeypoints(grayFrame,keypoints_scene, None)
    cv2.imshow("Real Time Cap sift", show_keypoints_scene)
    """
    
    try: # if there is no neighboor to pick two of them, this function will raise an error and we will catch it.
        knn_matches = matcher.knnMatch(descriptors_object, descriptors_scene, k=2) # we pick 2 nearest neighboor ,that is best matches ,for each descriptor from a query set(descriptors_object)
    except cv2.error as e:
        #print(e)
        if ((descriptors_object is None) or (descriptors_scene is None)):
            print("ERROR: No Descriptor!")
        
    # Filter matches using the Lowe's ratio test
    ratio_thresh = 0.5
    good_matches = []
    for m,n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    # Draw matches
    #img_matches = np.empty((max(img_object.shape[0], img_object.shape[0]), img_object.shape[1]+img_object.shape[1], 3), dtype=np.uint8)
    
    #result = cv2.drawMatches(img_object, keypoints_object, grayFrame, keypoints_scene, good_matches, grayFrame,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    target_lock_radius = 100
    cv2.circle(frame,(int(frame_width/2),int(frame_height/2)), target_lock_radius, (0,255,0), 1,  cv2.LINE_AA) # draws a circle representing landing zone (if center of the targeted object lies in this circle then land etc.)

    # lines for left,right,up,down boundaries
    cv2.line(frame,(int(frame_width/2 +target_lock_radius),0),(int(frame_width/2 +target_lock_radius),int(frame_height)),(255,0,0),1)
    cv2.line(frame,(0,int(frame_height/2 + target_lock_radius)),(int(frame_width),int(frame_height/2 + target_lock_radius)),(255,0,0),1)
    cv2.line(frame,(int(frame_width/2 -target_lock_radius),0),(int(frame_width/2-target_lock_radius),int(frame_height)),(255,0,0),1)
    cv2.line(frame,(0,int(frame_height/2 - target_lock_radius)),(int(frame_width),int(frame_height/2 - target_lock_radius)),(255,0,0),1)
    
    # If we have enough good matches, draw boundaries of the targeted object
    min_match_count = 7
    queryPoints = np.float32([keypoints_object[i.queryIdx].pt for i in good_matches]).reshape(-1, 1, 2)   # extracting location of good matches from targeted image
    #print(queryPoints)
    if len(good_matches) >= min_match_count:
            
            trainPoints = np.float32([keypoints_scene[j.trainIdx].pt for j in good_matches]).reshape(-1, 1, 2) # extracting location of good matches from real time vision
            
            representativematrix, maskk = cv2.findHomography(queryPoints, trainPoints, cv2.RANSAC, 5.0)  # this matrix represents location of target in real time vision
            matchesMask = maskk.ravel().tolist()
            height, width = img_object.shape  # height and width of original targeted image
            points = np.float32([[0, 0],[0, height-1],[width-1, height-1],[width-1,0]]).reshape(-1, 1, 2)

            if (representativematrix is not None) : # if representativematrix is none then there is nothing to apply perspective Transform so following line will raise an exception((-215:Assertion failed) scn + 1 == m.cols in function 'cv::perspectiveTransform').
                adaptiveTemplate = cv2.perspectiveTransform(points, representativematrix)  # points will adapt matrix
                homography = cv2.polylines(frame, [np.int32(adaptiveTemplate)], True, (0,0,255), 3, cv2.LINE_AA)
                (x,y),radius = cv2.minEnclosingCircle(np.int32(adaptiveTemplate))  # finds center point and radius of minimum circle that encloses our location of good matches
                #print(x,y,radius)
                
                center_detected_object = (int(x),int(y))
                #print(type( center_detected_object))
                #radius = int(radius)
                img = cv2.circle(homography,center_detected_object,10,(255,0,0),-1,cv2.LINE_AA) 
                cv2.putText(homography,"("+str(int(x))+","+str(int(y))+")",(int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, .8,(0,0,255),2,cv2.LINE_AA) # displays estimation of coordinates of center of object by using matched points
                cv2.putText(homography,'OBJECT DETECTED',(120,15), cv2.FONT_HERSHEY_SIMPLEX, .6,(0,0,255),2,cv2.LINE_AA) #displays some text 
                #cv2.imshow("Homograpyh", frame)
                
                # line 126-161 :draws a transparent rectangle to the zone, where our targeted object's center point lies.
                if(center_detected_object[0] <= int(frame_width/2 -target_lock_radius) and center_detected_object[1] <= int(frame_height/2 - target_lock_radius)):
                    #print("ZONE 1")
                    #To draw a rectangle, you need top-left corner and bottom-right corner of rectangle. 
                    cv2.rectangle(blank,(0,0),(int(frame_width/2 -target_lock_radius),int(frame_height/2 - target_lock_radius)),(0,255,0),cv2.FILLED)
                    
                elif(center_detected_object[0] <= int(frame_width/2 -target_lock_radius) and center_detected_object[1] >= int(frame_height/2 - target_lock_radius) and center_detected_object[1] <= int(frame_height/2 + target_lock_radius)):
                    #print("ZONE 2")
                    cv2.rectangle(blank,(0,int(frame_height/2 - target_lock_radius)),(int(frame_width/2 -target_lock_radius),int(frame_height/2 + target_lock_radius)),(0,255,0),cv2.FILLED)
                elif(center_detected_object[0] <= int(frame_width/2 -target_lock_radius) and center_detected_object[1] >= int(frame_height/2 + target_lock_radius)):
                    #print("ZONE 3")
                    cv2.rectangle(blank,(0,int(frame_height/2 + target_lock_radius)),(int(frame_width/2 -target_lock_radius),int(frame_height)),(0,255,0),cv2.FILLED)
                elif(center_detected_object[0] >= int(frame_width/2 -target_lock_radius) and center_detected_object[0] <= int(frame_width/2 +target_lock_radius) and center_detected_object[1] <= int(frame_height/2 - target_lock_radius)  ):
                    #print("ZONE 4")
                    cv2.rectangle(blank,(int(frame_width/2 -target_lock_radius),0),(int(frame_width/2 +target_lock_radius),int(frame_height/2 - target_lock_radius)),(0,255,0),cv2.FILLED)
                elif(center_detected_object[0] >= int(frame_width/2 -target_lock_radius) and center_detected_object[0] <= int(frame_width/2 +target_lock_radius) and center_detected_object[1] >= int(frame_height/2 - target_lock_radius) and center_detected_object[1] <= int(frame_height/2 + target_lock_radius)):
                    #print("ZONE 5")
                    cv2.rectangle(blank,(int(frame_width/2 -target_lock_radius),int(frame_height/2 - target_lock_radius)),(int(frame_width/2 + target_lock_radius),int(frame_height/2 + target_lock_radius)),(0,255,0),cv2.FILLED)
                elif(center_detected_object[0] >= int(frame_width/2 -target_lock_radius) and center_detected_object[0] <= int(frame_width/2 +target_lock_radius) and center_detected_object[1] >= int(frame_height/2 + target_lock_radius)  ):
                    #print("ZONE 6")
                    cv2.rectangle(blank,(int(frame_width/2 -target_lock_radius),int(frame_height/2 + target_lock_radius)),(int(frame_width/2 +target_lock_radius),int(frame_height)),(0,255,0),cv2.FILLED)
                elif(center_detected_object[0] >= int(frame_width/2 +target_lock_radius) and center_detected_object[1] <= int(frame_height/2 - target_lock_radius)):
                    #print("ZONE 7")
                    cv2.rectangle(blank,(int(frame_width/2 +target_lock_radius),0),(int(frame_width),int(frame_height/2 - target_lock_radius)),(0,255,0),cv2.FILLED)
                elif(center_detected_object[0] >= int(frame_width/2 +target_lock_radius) and center_detected_object[1] >= int(frame_height/2 - target_lock_radius) and center_detected_object[1] <= int(frame_height/2 + target_lock_radius)):
                    #print("ZONE 8")
                    cv2.rectangle(blank,(int(frame_width/2 +target_lock_radius),int(frame_height/2 - target_lock_radius)),(int(frame_width),int(frame_height/2 + target_lock_radius)),(0,255,0),cv2.FILLED)
                elif(center_detected_object[0] >= int(frame_width/2 +target_lock_radius) and center_detected_object[1] >= int(frame_height/2 + target_lock_radius)):
                    #print("ZONE 9")
                    cv2.rectangle(blank,(int(frame_width/2 +target_lock_radius),int(frame_height/2 + target_lock_radius)),(int(frame_width),int(frame_height)),(0,255,0),cv2.FILLED)
                
                alpha = 0.4
                beta = (1.0 - alpha)
                cv2.addWeighted(blank, alpha, frame, beta, 0.0,frame) # to make rectangle transparent
                plt.plot(x,y)

           
    else:
        #print( "Not enough good matches are found - {}/{}".format(len(good_matches), min_match_count) )
        matchesMask = None

    draw_params = dict(matchColor = (255,0,255), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
    result = cv2.drawMatches(img_object, keypoints_object, frame, keypoints_scene, good_matches,None,**draw_params)
    
    
    cv2.imshow('SEEK', result)# displays result
    plt.show()
    if cv2.waitKey(1) == 27:   #press esc to quit
        break
cv2.destroyAllWindows()
cap.release()
