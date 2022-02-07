#-*- coding: utf-8 -*-
import cv2, math, time
import numpy as np

class ArucoSingleTracker():
    def __init__(self, marker_id, marker_size, camera_matrix, dist_coeffs, camera_no = 0, camera_size= [640,480], show_feed = True):
        self.marker_id = marker_id
        self.marker_size = marker_size
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.camera_no = camera_no
        self.show_feed = show_feed

        # define Aruco dictionary and parameters
        self.aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
        # detector parameters: https://docs.opencv.org/3.4/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_params.errorCorrectionRate = 0.6 # default 0.6

        self.cap = cv2.VideoCapture(self.camera_no)
        #Set the camera size as the one it was calibrated with
        #self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_size[0])
        #self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_size[1])
         
    
    def RotationMatrixToEulerAngles(self,R) :
        # Checks if a matrix is a valid rotation matrix.
        def isRotationMatrix(R) :
            Rt = np.transpose(R)
            shouldBeIdentity = np.dot(Rt, R)
            I = np.identity(3, dtype = R.dtype)
            n = np.linalg.norm(I - shouldBeIdentity)
            return n < 1e-6

        assert(isRotationMatrix(R))
        
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        
        singular = sy < 1e-6

        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

        return np.array([x, y, z])

    def AbortMission(self):
        self.Abort = True
    def Seek(self, is_loop = True):
        
        self.Abort = False
        marker_found = False
        x_marker,y_marker,z_marker,roll_camera,pitch_camera = "N/A ","N/A ","N/A ","N/A ","N/A "
        frame_width  = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
        frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
        # 180 degree rotation matrix around the z axis
        R_flip_z  = np.zeros((3,3), dtype=np.float32)
        R_flip_z[0,0] =-1.0
        R_flip_z[1,1] =-1.0
        R_flip_z[2,2] =1.0
        R_flip_x  = np.zeros((3,3), dtype=np.float32)
        R_flip_x[0,0] =1.0
        R_flip_x[1,1] =-1.0
        R_flip_x[2,2] =-1.0

        prev_frame_time,new_frame_time = 0,0
        while not self.Abort:
            if self.cap.isOpened():
                ret , frame = self.cap.read()
                #cap.read() returns a bool (True/False). If frame is read correctly, it will be True. So you can check end of the video by checking this return value.
                
            else:
                ret = False

            while ret:
                ret , frame = self.cap.read()
                
                blur = cv2.GaussianBlur(frame,(3,3),0) # blured (filtered) image with a 5x5 gaussian kernel to remove the noise
                gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)  # apply grayscale
                if  is_loop: 
                    # Calculating the fps 
                    new_frame_time = time.time() 
                    fps = 1/(new_frame_time-prev_frame_time) 
                    prev_frame_time = new_frame_time 
                    cv2.putText(frame,"fps: "+str(int(fps)),(15,15), cv2.FONT_HERSHEY_SIMPLEX, .6,(0,60,0),1,cv2.LINE_AA) # displays fps

                cv2.circle(frame,(int(frame_width/2),int(frame_height/2)), 10, (0,255,0), 1,  cv2.LINE_AA)
                # cv2.aruco.detectMarkers(  image, dictionary[, corners[, ids[, parameters[, rejectedImgPoints[, cameraMatrix[, distCoeff]]]]]] )
                (corners, ids, rejected) = cv2.aruco.detectMarkers(image=gray, dictionary=self.aruco_dict, parameters=self.aruco_params)
                if(len(corners) > 0 ): # Returns True if at least one ArUco marker is detected.
                    ids = ids.flatten() # Return a copy of the ids array collapsed into one dimension.
                    #print("ids: ",ids)
                    try:
                        # Draw detected markers in image.
                        # image =   cv.aruco.drawDetectedMarkers(   image, corners[, ids[, borderColor]]    )
                        cv2.aruco.drawDetectedMarkers(frame, corners, ids, borderColor = (0, 0, 255) )  # Draw A square around the markers
                    except cv2.error as e:
                        print(e)
                    if(self.marker_id in ids) :
                        marker_index = np.where(ids == self.marker_id)[0][0]
                        #print("marker_index: ",marker_index)
                        marker_found = True
                        #print(" marker_{} found!".format(self.marker_id))
                        # Pose estimation for single marker.
                        # rvecs, tvecs, _objPoints  =   cv.aruco.estimatePoseSingleMarkers( corners, markerLength, cameraMatrix, distCoeffs[, rvecs[, tvecs[, _objPoints]]] )
                        # markerLength  the length of the markers' side. The returning translation vectors will be in the same unit. Normally, unit is meters.
                        rvec, tvec= cv2.aruco.estimatePoseSingleMarkers(corners=corners[marker_index], markerLength=self.marker_size, cameraMatrix=self.camera_matrix, distCoeffs=self.dist_coeffs)
                        #print("rvec: ",rvec)
                        #print("tvec: ",tvec)
                        # The translation vector contains the marker position relative to the camera.
                        translation_mtx = np.matrix(tvec)
                        #print("translation_mtx: ",translation_mtx)
                        translation_mtx_T = np.transpose(translation_mtx)
                        # Convert the rotation vector to the rotation matrix, which contains the roll, pitch, and yaw angle of the marker relative to the camera in marker’s coordinate system.
                        # dst, jacobian	=	cv.Rodrigues(	src[, dst[, jacobian]]	)
                        rotation_mtx = np.matrix(cv2.Rodrigues(rvec)[0]) # Converts a rotation vector to a rotation matrix. 
                        #print("rotation_mtx: ",rotation_mtx)
                        rotation_mtx_T = np.transpose(rotation_mtx)
                        # Calculate the camera position with respect to the marker by multiplying the transpose of the rotation matrix and the translation matrix.
                        cam_position = -rotation_mtx_T * translation_mtx_T   # both matrices are 3x3
                        x_marker,y_marker,z_marker = tvec[0,0,:]
                        #print("cam_position: ",cam_position)

                        marker_corners = corners[marker_index].reshape((4, 2))
                        (topLeft, topRight, bottomRight, bottomLeft) = marker_corners
                        # convert each of the (x, y)-coordinate pairs to integers
                        topRight = (int(topRight[0]), int(topRight[1]))
                        bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                        bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                        topLeft = (int(topLeft[0]), int(topLeft[1]))
                        # draw the ArUco marker ID on the frame
                        cv2.putText(frame, str(self.marker_id),(topLeft[0] +15 , topLeft[1] - 15),cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 0, 0), 2)

                        cv2.putText(frame, "Camera position wrt target(meters): x=%4.2f  y=%4.2f  z=%4.2f"%(cam_position[0], cam_position[1], cam_position[2]),
                        (5, 30),cv2.FONT_HERSHEY_SIMPLEX, .5,(0,0,255),1,cv2.LINE_AA)
                        cv2.putText(frame, "Target position wrt camera frame(meters): x=%4.2f  y=%4.2f  z=%4.2f"%(x_marker, y_marker, z_marker),(5, 45),cv2.FONT_HERSHEY_SIMPLEX, .5,(0,0,255),1,cv2.LINE_AA)
                        # Calculate the attitude of the camera regarding the marker by multiplying a 180-degree rotation matrix around the z-axis of the camera coordinate system to the rotationmatrix
                        altitude_cam_wrt_marker = rotation_mtx * R_flip_x
                        #print("altitude_cam_wrt_marker: ",altitude_cam_wrt_marker)
                        roll_camera, pitch_camera, yaw_camera = self.RotationMatrixToEulerAngles(altitude_cam_wrt_marker)
                        roll_camera = math.degrees(roll_camera)
                        pitch_camera = math.degrees(pitch_camera)
                        yaw_camera = math.degrees(yaw_camera)
                        cv2.putText(frame, "Camera principal axis wrt target(degree): roll=%4.2f  pitch=%4.2f  yaw=%4.2f"%(roll_camera,pitch_camera,yaw_camera), 
                        (5, 60),cv2.FONT_HERSHEY_SIMPLEX, .4,(0,0,255),1,cv2.LINE_AA)
                        
                        cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, (0.5)*self.marker_size)  # Draw the x,y,z axes
                        
                    else:
                        marker_found = False
                        print("Searching for markerid:{}...".format(self.marker_id))

                if (self.show_feed):
                    cv2.imshow("Real-time frame", frame)
                    
                    if cv2.waitKey(1) == 27:   #press esc to quit
                        self.cap.release()
                        cv2.destroyAllWindows()
                        break
                if not is_loop: 
                    
                    return(marker_found, x_marker,y_marker,z_marker,roll_camera,pitch_camera)
                    
                    
                
if __name__ == "__main__":
    print("running") 
    camera_matrix  = np.array( [ [ 6.0784457614025803e+02          ,            0.         ,   3.2306280294111639e+02],
                                 [           0.                    , 6.0776688471513398e+02,   2.3031092286690938e+02],
                                 [           0.                    ,            0.         ,             1.          ]])
    dist_coeffs  = np.array( [ 2.2848729690411801e-01, -9.5226223066766025e-01, 9.7446890329030999e-04, 8.5096039022205565e-05, 1.3294917618602062e-01 ])                                     
    tracker = ArucoSingleTracker(marker_id=4, marker_size=0.190, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    tracker.Seek()
            
