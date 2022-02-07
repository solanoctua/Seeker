import math
import cv2
import glob
import numpy as np 

# https://stackoverflow.com/questions/50857278/raspicam-fisheye-calibration-with-opencv
# Checkboard dimensions
CHECKERBOARD = (6,9)
edge_of_square = 24 # 24 milimeters = 0.024 meters

path_of_images = "C:/Users/asus/Desktop/Samurai-Seeker/FisheyeCalibration"
filename = path_of_images +"/*.jpg"

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_FIX_SKEW + cv2.fisheye.CALIB_CHECK_COND 

w = 640
h = 480
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
#objp = np.zeros((6*9,3), np.float32)
#bjp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)  
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) # creates a matrix containing distance of upper-left corner of every square from origin taking the most upper-left corner as origin
#[[[0. 0. 0.][1. 0. 0.][2. 0. 0.][3. 0. 0.][4. 0. 0.][5. 0. 0.][0. 1. 0.][1. 1. 0.][2. 1. 0.]....]]  

objp = objp * edge_of_square # to find location of corners in real world we multiply the matrix with edge_of_square

# Arrays to store object points and image points from all the images.                    
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


# We will for loop calibration images
images = glob.glob(filename)
numberofimgs = 0
for fname in images:
    distorted_img = cv2.imread(fname) # read images one by one
    numberofimgs += 1
    #print(fname)
    shape = distorted_img.shape[:2][::-1]
    dimensions = distorted_img.shape
    print(dimensions)
    gray = cv2.cvtColor(distorted_img, cv2.COLOR_BGR2GRAY) # apply BGR to Gray conversion
    ret, corners = cv2.findChessboardCorners(gray, (6,9),cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE )# Finds the positions of internal corners of the chessboard, if desired number of corners are found in the image then ret = true 
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners, (3,3), (-1,-1), subpix_criteria) #The cornerSubPix function iterates to find the sub-pixel accurate location of corners.
        imgpoints.append(corners)
        # Draw and display the corners
        """
        cv2.drawChessboardCorners(distorted_img, (9,6), corners, ret)
        cv2.putText(distorted_img,"name of the file:"+str(fname),(15,15), cv2.FONT_HERSHEY_SIMPLEX, .4,(0,0,255),1,cv2.LINE_AA) #displays some text
        cv2.imshow('distorted_img', distorted_img)
       
        cv2.waitKey(100)

cv2.destroyAllWindows()
"""
# Camera matrix(3x3 matrix),Distortion coefficients(1x4 matrix),Rotation Vectors,Translation Vectors
mtx = np.zeros((3, 3))
dist = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(numberofimgs)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(numberofimgs)]
#print("objectpoints: ",objpoints)
#print("imgpoints: ",imgpoints)
# https://docs.opencv.org/3.4.15/db/d58/group__calib3d__fisheye.html#gad626a78de2b1dae7489e152a5a5a89e1

#shape is a tuple of (row (height), column (width), color (3)) we only need row and column
ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    mtx,
    dist,
    rvecs,
    tvecs,
    calibration_flags,
    (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
print("dist: ",dist)
print("mtx: ",mtx)
# Save mtx and dist values into a txt file

with open(path_of_images+"/fisheyecalibration_data.txt", "a") as file:
    np.savetxt(file, mtx, delimiter=',', header="mtx: ",)
    np.savetxt(path_of_images+"/fisheyecalibration_data.txt", dist, delimiter=',', header="dist: ",)
    file.close()

file = cv2.FileStorage(path_of_images+"/fisheyecalibration_data_opencv.txt", cv2.FILE_STORAGE_WRITE)
file.write("mtx", mtx)
file.write("dist", dist)
file.release()
print("camera matrix and distortion coefficients are saved")

#img = path_of_images +"/*1.jpg"
img = cv2.imread(path_of_images+"/0.jpg")
cv2.imshow("img", img)
"""
new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
undistorted_img = cv2.undistort(img, mtx, dist, None, new_mtx)
"""

map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, size=(640,480), m1type= cv2.CV_16SC2)  #cv2.CV_16SC2
undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT ) #

"""
scaled_K = mtx * 2
scaled_K[2][2] = 1.0  
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, dist, (640,480), np.eye(3), balance=0.8)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, dist, np.eye(3),new_K, (640,480), cv2.CV_16SC2)
undist_image = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
"""
cv2.imshow("undistorted image", undistorted_img)
cv2.waitKey(50000)
key = cv2.waitKey(1)
if key == 27:   #press esc to quit
    cv2.destroyAllWindows()    



