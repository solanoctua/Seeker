import numpy as np
import cv2, os, glob
from pathlib import Path

# Get the current working directory 
cwd = os.getcwd() 
path_of_images = cwd
#path_of_images = "C:/Users/asus/Desktop/Samurai-Seeker/Calibration_images/x"

cap = cv2.VideoCapture(0)

frame_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
print("frame_width: ",frame_width)
print("frame_height: ",frame_height)

# FINDING CHESSBOARD CORNERS
edge_of_square = 0.024 # in meters
# termination criteria:
# cv.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached.
# cv.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
# cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met.
# maxCount	The maximum number of iterations or elements to compute.
# epsilon - Required accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)   
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0)                     
objp = np.zeros((6*9,3), np.float32) # create 6x9 matrix ,which each entry will(currently all 0) contain (x,y,z) coordinates of corners of chessboard in real(3D) world                                                  
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)      
objp = objp * edge_of_square # to find location of corners in real world we multiply the matrix with edge_of_square
# Arrays to store object points and image points from all the images.                    
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
axis = np.float32([[edge_of_square*3,0,0], [0,edge_of_square*3,0], [0,0,-3*edge_of_square]]).reshape(-1,3)
def draw(img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img
    
def create_output_path(output_path):
    output_path = Path(output_path)
    # Create the output paths if it do not exist.
    output_path.mkdir(parents=True, exist_ok=True)
    """
    if not os.path.exists( output_path):
        os.mkdir(output_path)
        
        print("Directory ", output_path,  " Created")
    else:    
        print("Directory ", output_path, " already exists")
    """
def findCorners(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # apply BGR to Gray conversion
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)# Finds the positions of internal corners of the chessboard, if desired number of corners are found in the image then ret = true 
        # If found, add object points, image points (after refining them)
        if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria) #The cornerSubPix function iterates to find the sub-pixel accurate location of corners.
                imgpoints.append(corners)
                # Draw and display the corners
                cv2.drawChessboardCorners(frame, (9,6), corners2, ret)
                cv2.imshow("chessboard corners", frame)
        else:
            print("No chessboard in the frame!")
        return image
        
count = 0
if cap.isOpened():
    ret,frame = cap.read()
    create_output_path(cwd +"/results")
else:
    ret = False
while ret:
    ret , frame = cap.read()
    clone = frame.copy()
    cv2.putText(frame,"press 'esc' to quit",(15,15), cv2.FONT_HERSHEY_SIMPLEX, .6,(0,0,255),1,cv2.LINE_AA) #displays some text
    cv2.putText(frame,"press '+' to capture the image",(15,35), cv2.FONT_HERSHEY_SIMPLEX, .6,(255,0,0),1,cv2.LINE_AA) #displays some text
    cv2.putText(frame,"press '-' to discard the image",(15,55), cv2.FONT_HERSHEY_SIMPLEX, .6,(255,0,0),1,cv2.LINE_AA) #displays some text 
    cv2.putText(frame,"press 'r' to start calibration",(15,75), cv2.FONT_HERSHEY_SIMPLEX, .6,(255,0,0),1,cv2.LINE_AA) #displays some text 
    cv2.imshow("camera",frame)
    key = cv2.waitKey(1) 
    if key == 27:   #press esc to quit
        print("Quiting..")
        cap.release()
        break
    
    elif key == ord("+"):
        count += 1
        print("Saving image to ",path_of_images)
        #cv2.imwrite("/{}{}.jpg".format(path_of_images+"/",count),clone)
        cv2.imwrite(path_of_images +"/"+str(count)+".jpg", clone)
        print("Calculating corners..")
        cv2.imwrite("{}{}.jpg".format(path_of_images+"/results/corners_",count), findCorners(frame))

    elif key == ord("-"):
        try:
          print("Erasing last image")      
          os.remove(path_of_images +"/"+str(count)+".jpg")
          os.remove(path_of_images +"/results/corners_"+str(count)+".jpg")
        except:
          print("No image to discard")
        
    elif key == ord("r"): # START CALIBRATION WITH SAVED IMAGES
        print("Calibrating..")
        filename = path_of_images +"/*.jpg"
        images = glob.glob(filename)
        for fname in images:
            distorted_img = cv2.imread(fname) # read images one by one
            gray = cv2.cvtColor(distorted_img, cv2.COLOR_BGR2GRAY) # apply BGR to Gray conversion
            ret, corners = cv2.findChessboardCorners(gray, (9,6), None)# Finds the positions of internal corners of the chessboard, if desired number of corners are found in the image then ret = true 
                # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        # Camera matrix(3x3 matrix),Distortion coefficients(1x5 matrix),Rotation Vectors,Translation Vectors
        size = (int(frame_width),int(frame_height))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, size, None, None) #Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.
                                                                             #shape is a tuple of (row (height), column (width), color (3)) we only need row and column
        # Save mtx and dist values into a txt file
        with open(path_of_images+"/calibration_data.txt", "a") as file:
            np.savetxt(file, mtx, delimiter=',', header="mtx: ",)
            np.savetxt(file, dist, delimiter=',', header="dist: ",)
            file.close()
        print("camera matrix and distortion coefficients are saved")

        # Calculate Re-projection error 
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
            total_error += error
        print ("Re-projection error: ", total_error/len(objpoints))

        # undistort one of distorted images
        """
        distorted_img = cv2.imread(path_of_images+"/1.jpg")
        h,  w = distorted_img.shape[:2]
        new_camera_mtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))  # Returns the new camera intrinsic matrix 
        dst = cv2.undistort(distorted_img, mtx, dist, None, new_camera_mtx) # The undistort function transforms an image to compensate radial and tangential lens distortion
        # crop the image
        x,y,w,h = roi
        dst = dst[y:y+h, x:x+w]
        cv2.imwrite(path_of_images +"/undistorted_sample_0.jpg",dst)
        
        filename = path_of_images +"/*.jpg"
        """
        # Draw poses
        images = glob.glob(filename)
        count = 1
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

            if ret == True:
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

                # Find the rotation and translation vectors.
                _ ,rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)
            
                # project 3D points to image plane
                imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist) #Projects 3D points(axis in this case) to an image plane.
                # Draw and display chessboard poses
                img_wpose = draw(img,corners2,imgpts)
                cv2.imshow('img',img_wpose)
                
                cv2.imwrite(path_of_images +"/results/wpose_"+str(count)+".jpg", img_wpose)
                cv2.waitKey(100)
                count += 1
        print("images with poses are saved")
cv2.destroyAllWindows()
cap.release()
