import numpy as np
import cv2 
import glob
import os

# https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html#calibration
# http://www.bim-times.com/opencv/3.3.0/d9/d0c/group__calib3d.html#ga3207604e4b1a1758aa66acb6ed5aa65d
edge_of_square = 24 # 24 milimeters = 0.024 meters

# termination criteria:
# cv.TERM_CRITERIA_EPS - stop the algorithm iteration if specified accuracy, epsilon, is reached.
# cv.TERM_CRITERIA_MAX_ITER - stop the algorithm after the specified number of iterations, max_iter.
# cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER - stop the iteration when any of the above condition is met.
# maxCount	The maximum number of iterations or elements to compute.
# epsilon - Required accuracy
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)   

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0)                     
objp = np.zeros((6*9,3), np.float32) # create 6x9 matrix ,which each entry will(currently all 0) contain (x,y,z) coordinates of corners of chessboard in real(3D) world 
#print(objp)                                                 
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)      
#print(objp)
objp = objp * edge_of_square # to find location of corners in real world we multiply the matrix with edge_of_square
#print(objp)
# Arrays to store object points and image points from all the images.                    
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


path_of_images = "C:/Users/asus/Desktop/Samurai-Seeker/Calibration_images"
filename = path_of_images +"/*.jpg"

#print("{} image found.".format(len(os.listdir(path_of_images))))
# We will for loop calibration images
images = glob.glob(filename)
for fname in images:
    distorted_img = cv2.imread(fname) # read images one by one
    gray = cv2.cvtColor(distorted_img, cv2.COLOR_BGR2GRAY) # apply BGR to Gray conversion
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)# Finds the positions of internal corners of the chessboard, if desired number of corners are found in the image then ret = true 
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria) #The cornerSubPix function iterates to find the sub-pixel accurate location of corners.
        imgpoints.append(corners)
        # Draw and display the corners
        """
        cv2.drawChessboardCorners(distorted_img, (9,6), corners2, ret)
        
        cv2.imshow('distorted_img', distorted_img)
        cv2.waitKey(500)
        """
        
cv2.destroyAllWindows()
# Camera matrix(3x3 matrix),Distortion coefficients(1x5 matrix),Rotation Vectors,Translation Vectors
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) #Finds the camera intrinsic and extrinsic parameters from several views of a calibration pattern.
                                                                         #shape is a tuple of (row (height), column (width), color (3)) we only need row and column
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

# Calculate Re-projection error 
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    total_error += error
print ("Re-projection error: ", total_error/len(objpoints))

#**********************************************Sample of undistorted image********************************************
# undistort one of distorted images
distorted_img = cv2.imread(path_of_images+"/0.jpg")
h,  w = distorted_img.shape[:2]
new_camera_mtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))  # Returns the new camera intrinsic matrix 

dst = cv2.undistort(distorted_img, mtx, dist, None, new_camera_mtx) # The undistort function transforms an image to compensate radial and tangential lens distortion

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite(path_of_images +"/calibration_result_0.jpg",dst)

#***********************************************Pose Estimation******************************************************
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
objp = objp * edge_of_square # to find location of corners in real world we multiply the matrix with edge_of_square

axis = np.float32([[72,0,0], [0,72,0], [0,0,-72]]).reshape(-1,3)
count = 0
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
    
        img_wpose = draw(img,corners2,imgpts)
        cv2.imshow('img',img_wpose)
        cv2.imwrite(path_of_images +"/"+"images_wposes/"+"wpose_"+str(count)+".jpg", img_wpose)
        cv2.waitKey(500)
        count+=1
print("images with poses are saved")    
cv2.destroyAllWindows()


