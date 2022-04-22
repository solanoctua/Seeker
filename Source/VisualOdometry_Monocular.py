import numpy as np
import matplotlib.pyplot as plt
import cv2, time # cv2.__version__ = '4.0.0'

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.colors as mcol
import matplotlib.cm as cm
import matplotlib.animation as animation

class VisualOdometryMono():
    def __init__(self, *args, **kwargs):
        self.visuals = 1
        self.record = 1
        # CAMERA PARAMETERS
        self.cam = cv2.VideoCapture(0)  # cv2.VideoCapture("C:/Users/asus/Desktop/Seeker/Seeker/TargetImages/Sample7.mp4") 
        self.frame_width, self.frame_height = (2160//4, 4096//16)## (340,340) # #  # 1280,720
        """
        self.CameraMatrix = np.array( [ [8.6297319278294219e+02,          0.          , 3.2410970150269952e+02], 
                                        [           0.         ,8.6289007075149686e+02, 2.3151809145852621e+02],
                                        [           0.         ,          0.          ,            1.         ]])
        """
         # calib for seqn 00 
        self.CameraMatrix = np.array( [ [7.188560000000e+02, 0.                , 6.071928000000e+02], 
                                        [0.                , 7.188560000000e+02, 1.852157000000e+02],
                                        [0.                , 0.                , 1.                ]])
        
        """
        # calib for seqn 01 
        self.CameraMatrix = np.array( [ [7.070912000000e+02, 0.                , 6.018873000000e+02], 
                                        [0.                , 7.070912000000e+02, 1.831104000000e+02],
                                        [0.                , 0.                , 1.                ]])
        """   
        """
        # calib for seqn 06                              
        self.CameraMatrix = np.array( [ [7.070912000000e+02, 0.000000000000e+00, 6.018873000000e+02], 
                                        [0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02],
                                        [0.                , 0.                , 1.                ]])                                
                                         
        """                                
        self.DistortionCoefficients = np.array( [ 1.3876171595109568e-01, -5.0495233579131149e-01, -1.4364550355797534e-03, 3.0938437583063767e-03, 1.3034844698951493e+00 ])
        
        self.FeatureDetector = cv2.ORB_create(nfeatures= 3000) # https://docs.opencv.org/4.0.0/db/d95/classcv_1_1ORB.html
        # Flann parameters for ORB
        indexparameters= dict(algorithm = 6, 
                              table_number = 12,# 6, # 12
                              key_size = 20,# 12,     # 20
                              multi_probe_level = 1)#1) #2

        searchparameters = dict(checks=50) #30
        self.Matcher = cv2.FlannBasedMatcher(indexparameters, searchparameters) #https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
        self.GoodMatchesRatio = 0.8   #0.75 For Lowe's ratio test
        self.Frames = []

        np.set_printoptions(suppress= True)

    def computeKeyandDes(self, frame):
        #print("computing keypoints and descriptors..")
        """
        features = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), mask = None, **goodFeatureParams)
        keypoints = [cv2.KeyPoint(x = feature[0][0], y = feature[0][1], _size = 20) for feature in features ]  # size: diameter of the meaningful keypoint neighborhood
        keypoints, descriptors = self.FeatureDetector.compute(frame, keypoints)
        """
        # FEATURE DETECTION
        keypoints, descriptors = self.FeatureDetector.detectAndCompute(frame, None) #https://docs.opencv.org/4.x/d2/d29/classcv_1_1KeyPoint.html
        """
        print("keypoints = ",keypoints)
        print("descriptors.shape() : ",descriptors.shape)
        print("descriptors = ",descriptors)
        """
        return keypoints, descriptors

    def findMatches(self, frame_number, frame):
        #print("finding matches..")
        # FEATURE DETECTION
        old_keypoints, old_descriptors = self.computeKeyandDes(self.Frames[frame_number - 1])
        new_keypoints, new_descriptors = self.computeKeyandDes(self.Frames[frame_number]) 
        # FEATURE MATCHING/TRACKING
        goodMatches = []
        try: # if there is no neighboor to pick two of them, this function will raise an error and we will catch it.
            matches = self.Matcher.knnMatch(old_descriptors, new_descriptors, k=2)#https://docs.opencv.org/4.x/d4/de0/classcv_1_1DMatch.html#ab9800c265dcb748a28aa1a2d4b45eea4
        except cv2.error as e:
            print(e)
            
            if ((old_descriptors is None) or (new_descriptors is None)):
                print("ERROR: No Descriptor!")

        #print("matches: ",matches)
        for i,pair in enumerate(matches):
            try:
                m, n = pair
            except:
                continue
            if m.distance < self.GoodMatchesRatio * n.distance:  # Lowe's ratio test
                goodMatches.append(m)
                """
                srcPoints.append(old_keypoints[m.queryIdx].pt) # points in the new frame
                dstPoints.append(new_keypoints[m.trainIdx].pt) # points in the old frame
                """
        """        
        print("len(goodMatches): ",len(goodMatches))       
        print("len(old_keypoints): ",len(old_keypoints))
        print("len(new_keypoints): ",len(new_keypoints))
        """
        srcPoints = np.float32([old_keypoints[m.queryIdx].pt for m in goodMatches])#.reshape(-1, 1, 2) # points in the old frame 
        dstPoints = np.float32([new_keypoints[m.trainIdx].pt for m in goodMatches])#.reshape(-1, 1, 2) # points in the new frame
        #print("srcPoints: ",srcPoints)
        #print("goodMatches: ",goodMatches)
        drawmatches = None
        if(self.visuals == 1 and len(goodMatches)):
            """
            for m in goodMatches:
                x1,y1 = old_keypoints[m.queryIdx].pt
                x2,y2 = new_keypoints[m.trainIdx].pt
                #print("({},{}) <==> ({},{})".format(x1,y1,x2,y2))
                #drawmatches = cv2.line(frame, (round(x1),round(y1)), (round(x2),round(y2)), (255,0,0),2)
                drawmatches = cv2.circle(frame, (int(x1), int(y1)), 3, (0,0,255), 1)
            """
            
            draw_params = dict(matchColor = (255,0,155), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = None, # draw only inliers
                   flags = 2 # NOT_DRAW_SINGLE_POINTS 	
                   )
                   
            drawmatches = cv2.drawMatches(self.Frames[frame_number-1], old_keypoints,self.Frames[frame_number], new_keypoints,  goodMatches, None, **draw_params)
            
            cv2.imshow("Matches", drawmatches)
            
        return srcPoints, dstPoints, drawmatches
    @staticmethod
    def calculateTransformationMatrix(rotationMatrix, translationVector):
        #print("calculating transformation matrix..")
        transformationMatrix = np.eye(4, dtype= np.float64)
        transformationMatrix[:3,:3] = rotationMatrix
        transformationMatrix[:3,3] = translationVector.T
        #print("transformationMatrix =\n",transformationMatrix)
        return transformationMatrix

    def decomposeEssentialMatrix(self, EssentialMatrix, point_old, point_new):
        #print("decomposing essential matrix")
        # POSE ESTIMATION
        R1, R2, t = cv2.decomposeEssentialMat(EssentialMatrix)# https://docs.opencv.org/4.0.0/d9/d0c/group__calib3d.html#ga54a2f5b3f8aeaf6c76d4a31dece85d5d
        #print("R1 = {}\nR2 = {}\nt = {}".format(R1, R2, t))
        #print("np.ndarray.flatten(t): ",np.ndarray.flatten(t))
        T1 = self.calculateTransformationMatrix(R1, t) # 4x4 matrix
        T2 = self.calculateTransformationMatrix(R2, t)
        T3 = self.calculateTransformationMatrix(R1, -t)
        T4 = self.calculateTransformationMatrix(R2, -t)

        T = [T1, T2, T3, T4]
        #print("T = [T1, T2, T3, T4] = \n{} ".format(T))
        # Make cameraMatrix homogeneous
        
        cameraMatrix = np.concatenate((self.CameraMatrix, np.zeros((3,1))), axis= 1) # make last column of the cameraMatrix 0 0 0 so becomes 3x4 matrix
        possiblePoses = [cameraMatrix @ T1, cameraMatrix @ T2, cameraMatrix @ T3, cameraMatrix @ T4]
        """
        print("possiblePoses1 = \n",possiblePoses[0])
        print("possiblePoses2 = \n ",possiblePoses[1].tolist())
        print("possiblePoses3 = \n",possiblePoses[2])
        print("possiblePoses4 = \n",possiblePoses[3])
        """
        # We must take poses only in front of the camera
        pozitivePoses = []
        for P,T in zip(possiblePoses, T):  # poses are 3x4 matrices
            #print("Pose:",P)
            points3D_old = cv2.triangulatePoints(cameraMatrix, P, point_old.T, point_new.T)
            points3D_new = T @ points3D_old
            # dehomogenize
            points3D_old = points3D_old[:3, : ]/points3D_old[3, : ]
            points3D_new = points3D_new[:3, : ]/points3D_new[3, : ]
            #print("points3D_new: ",points3D_new)
            # Find the number of points there has positive z coordinate in both cameras
            sum_of_pos_z_old = sum(points3D_old[2, :] > 0)
            sum_of_pos_z_new = sum(points3D_new[2, :] > 0)

            #distance = sum(points3D_old[2, : ]>0) + sum(points3D_new[2, : ]>0)
            distance = sum_of_pos_z_old + sum_of_pos_z_new
            scale = np.mean(np.linalg.norm(points3D_old.T[:-1]-points3D_old.T[1:], axis = -1)/
                            np.linalg.norm(points3D_new.T[:-1]-points3D_new.T[1:], axis = -1))
            pozitivePoses.append(distance + scale)
        #print("pozitivePoses = ",pozitivePoses)
        max = np.argmax(pozitivePoses)
        if (max == 0):
            return R1, np.ndarray.flatten(t), points3D_old
        elif(max == 1):
            return R2, np.ndarray.flatten(t), points3D_old
        elif(max == 2):
            return R1, np.ndarray.flatten(-t), points3D_old
        elif(max == 3):
            return R2, np.ndarray.flatten(-t), points3D_old

    def estimatePose(self, oldPoints, newPoints):
        EssentialMatrix, mask = cv2.findEssentialMat(oldPoints, newPoints, self.CameraMatrix)# https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga0b166d41926a7793ab1c351dbaa9ffd4
        #print("EssentialMatrix: ",EssentialMatrix)
        R, t, points3D = self.decomposeEssentialMatrix(EssentialMatrix, oldPoints, newPoints)
        #print("R = {} t = {} ".format(R,t))
        return self.calculateTransformationMatrix(R, t),points3D

def main():
    np.set_printoptions(suppress=True)
    vo = VisualOdometryMono()
    
    numberOfFrames = 500
    plotSize = 200
    fig = plt.figure()
    fig.set_size_inches(14.5, 10.5, forward=True)
    ax = fig.add_subplot(111, projection='3d')
    plt.title("Best VSLAM ever")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Make a user-defined colormap.
    cm1 = mcol.LinearSegmentedColormap.from_list("RedToBlue",["r","b"])
    cm2 = mcol.LinearSegmentedColormap.from_list("GreenToYellow",["green","black"])
    # Make a normalizer that will map the time values from
    # [start_time,end_time+1] -> [0,1].
    cnorm = mcol.Normalize(1,numberOfFrames)

    # Turn these into an object that can be used to map time values to colors and
    # can be passed to plt.colorbar().
    cpick1 = cm.ScalarMappable(norm=cnorm,cmap=cm1)
    cpick1.set_array([])
    cpick2 = cm.ScalarMappable(norm=cnorm,cmap=cm2)
    cpick2.set_array([])
    prev_frame_time = 0
    new_frame_time = 0
    file = open("C:/Users/asus/Desktop/Odometry/sequence00EstimatedPoses.txt", "w")

    if vo.cam.isOpened():
        #ret,frame = vo.cam.read()
        ret = True
        frame = cv2.imread("C:/Users/asus/Desktop/Odometry/data_odometry_gray/00/image_0/000000.png",0)
        frame_number = 0
        vo.Frames.append(frame)
        """
        frame = cv2.resize(frame,(vo.frame_width, vo.frame_height ))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # apply grayscale
        vo.Frames.append(gray)
        """
        #output = cv2.VideoWriter("C:/Users/asus/Desktop/Seeker/MonocularVOMatches.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (vo.frame_width*2, vo.frame_height)) # 'M','J','P','G' #https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html
        
        currentPose = np.array([[1.000000e+00, 9.043680e-12, 2.326809e-11, 5.551115e-17 ],
                                [9.043683e-12, 1.000000e+00, 2.392370e-10, 3.330669e-16 ],
                                [2.326810e-11, 2.392370e-10, 9.999999e-01, -4.440892e-16],
                                [0           , 0           , 0           , 1            ]])
        
        """
        1.000000000000000000 0.000000000009043680 0.000000000023268090 0.000000000000000056
         0.000000000009043683 1.000000000000000000 0.000000000239237000 0.000000000000000333
          0.000000000023268100 0.000000000239237000 0.999999900000000053 -0.000000000000000444 
        """
        """
        currentPose = np.array([[1.000000000000000000, 0.000000000011976250, 0.000000000170463800, 0.000000000000000000 ],
                                [0.000000000011976250, 1.000000000000000000, 0.000000000356250300, -0.000000000000000111 ],
                                [0.000000000170463800, 0.000000000356250300, 1.000000000000000000, 0.000000000000000222],
                                [0                   , 0                   , 0                   , 1                    ]])
        """                         
        print("initialPose = \n",currentPose)
        z = 0
    else: 
        ret = False
    while ret :
        if len(vo.Frames) == 2:
            vo.Frames.pop(0)
            frame_number = -1
        
        z += 1
        #ret,frame = vo.cam.read()
        #print(str(z).rjust(6, '0'))
        frame = cv2.imread("C:/Users/asus/Desktop/Odometry/data_odometry_gray/00/image_0/{}.png".format(str(z).rjust(6, '0')),0)
        
        gray = frame
        """
        frame = cv2.resize(frame,(vo.frame_width, vo.frame_height ))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # apply grayscale
        """
        vo.Frames.append(gray)
        """
        print(len(vo.Frames))
        print("vo.Frames: \n",vo.Frames)
        """
        frame_number += 1
        
        #frame =cv2.flip(frame,-1)

        #Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        
        points2D_old, points2D_new, drawmatches = vo.findMatches(frame_number, gray)
        transformationMatrix, points3D_old = vo.estimatePose(points2D_old, points2D_new)
        #print("transformationMatrix = \n",transformationMatrix)
        #print("previousPose = \n",currentPose)
        currentPose = np.matmul(currentPose, np.linalg.inv(transformationMatrix))
        print("currentPose = \n",currentPose)
        print("currentPose (x,y,z) = ({},{},{})".format(currentPose[0,3],currentPose[2,3],currentPose[1,3]))
        
        
        file.write(str(currentPose[0,0])+" "+str(currentPose[0,1])+" "+ str(currentPose[0,2])+" "+ str(currentPose[0,3])+" "+
                str(currentPose[1,0])+" "+str(currentPose[1,1])+" "+ str(currentPose[1,2])+" "+ str(currentPose[1,3])+" "+
                str(currentPose[2,0])+" "+str(currentPose[2,1])+" "+ str(currentPose[2,2])+" "+ str(currentPose[2,3])+"\n")
        ax.scatter(currentPose[0,3],currentPose[2,3],0,marker=".",color=cpick1.to_rgba(z),s = plotSize )
        """
        Ys = points3D_old[:, 0]
        Zs = points3D_old[:, 1]
        Xs = points3D_old[:, 2]
        ax.scatter(Xs, Ys, Zs,marker ="1",color=cpick2.to_rgba(z))
        """
        if(vo.visuals == 1):
            cv2.putText(frame,"FPS:{}".format(int(fps)),(15,15),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1,cv2.LINE_AA)# Displays fps
            cv2.imshow("realTimeCamera", frame)
        """    
        if(vo.record == 1):
            
            #output.write(drawmatches)   
        """
        key=cv2.waitKey(1)
        if key==27 or z == numberOfFrames:   #press esc to quit
            break
    file.close()
    cv2.destroyAllWindows()
    vo.cam.release()
    #output.release()
    plt.colorbar(cpick1,label="Camera Pose Per Frame")
    plt.colorbar(cpick2,label="Point Cloud Per Frame")
    
    plt.savefig("C:/Users/asus/Desktop/Seeker/monocularVOseqn00.png")
    plt.show()


if __name__ == "__main__":
    
    main()