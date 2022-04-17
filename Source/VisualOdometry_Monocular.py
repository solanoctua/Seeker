import numpy as np
import matplotlib.pyplot as plt
import cv2, time # cv2.__version__ = '4.0.0'

class VisualOdometryMono():
    def __init__(self, *args, **kwargs):
        self.visuals = 1
        self.record = 0
        # CAMERA PARAMETERS
        self.cam = cv2.VideoCapture(0)
        self.frame_width, self.frame_height = (340,340)  # 1280,720
        self.CameraMatrix = np.array( [ [8.6297319278294219e+02,          0.          , 3.2410970150269952e+02], 
                                        [           0.         ,8.6289007075149686e+02, 2.3151809145852621e+02],
                                        [           0.         ,          0.          ,            1.         ] ])
        self.DistortionCoefficients = np.array( [ 1.3876171595109568e-01, -5.0495233579131149e-01, -1.4364550355797534e-03, 3.0938437583063767e-03, 1.3034844698951493e+00 ])
        self.FeatureDetector = cv2.ORB_create(nfeatures= 500) # https://docs.opencv.org/4.0.0/db/d95/classcv_1_1ORB.html
        # Flann parameters for ORB
        indexparameters= dict(algorithm = 6, 
                              table_number = 12,# 6, # 12
                              key_size = 20,# 12,     # 20
                              multi_probe_level = 2)#1) #2

        searchparameters = dict(checks=30) 
        self.Matcher = cv2.FlannBasedMatcher(indexparameters, searchparameters) #https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
        self.GoodMatchesRatio = 0.75  # For Lowe's ratio test
        self.Frames = []

        np.set_printoptions(suppress= True)

    def computeKeyandDes(self, frame):
        print("computing keypoints and descriptors..")
        """
        features = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), mask = None, **goodFeatureParams)
        keypoints = [cv2.KeyPoint(x = feature[0][0], y = feature[0][1], _size = 20) for feature in features ]  # size: diameter of the meaningful keypoint neighborhood
        keypoints, descriptors = self.FeatureDetector.compute(frame, keypoints)
        """
        keypoints, descriptors = self.FeatureDetector.detectAndCompute(frame, None) #https://docs.opencv.org/4.x/d2/d29/classcv_1_1KeyPoint.html
        print("keypoints = ",keypoints)
        print("descriptors.shape() : ",descriptors.shape)
        print("descriptors = ",descriptors)
        return keypoints, descriptors

    def findMatches(self, frame_number, frame):
        print("finding matches..")
        old_keypoints, old_descriptors = self.computeKeyandDes(self.Frames[frame_number - 1])
        new_keypoints, new_descriptors = self.computeKeyandDes(self.Frames[frame_number]) 

        goodMatches = []
        try: # if there is no neighboor to pick two of them, this function will raise an error and we will catch it.
            matches = self.Matcher.knnMatch(old_descriptors, new_descriptors, k=2)#https://docs.opencv.org/4.x/d4/de0/classcv_1_1DMatch.html#ab9800c265dcb748a28aa1a2d4b45eea4
        except cv2.error as e:
            print(e)
            if ((old_descriptors is None) or (new_descriptors is None)):
                print("ERROR: No Descriptor!")

        print("matches: ",matches)
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
        print("len(goodMatches): ",len(goodMatches))       
        print("len(old_keypoints): ",len(old_keypoints))
        print("len(new_keypoints): ",len(new_keypoints))
        
        srcPoints = np.float32([old_keypoints[m.queryIdx].pt for m in goodMatches])#.reshape(-1, 1, 2) # points in the old frame 
        dstPoints = np.float32([new_keypoints[m.trainIdx].pt for m in goodMatches])#.reshape(-1, 1, 2) # points in the new frame
        print("srcPoints: ",srcPoints)
        if(self.visuals == 1):
            """
            for m in goodMatches:
                x1,y1 = srcPoints[m.queryIdx].pt
                x2,y2 = dstPoints[m.queryIdx].pt
                cv2.line(frame, (round(x1),round(y1)), (round(x2),round(y2)), (255,0,0),2)
                cv2.circle(frame, (int(x1), int(y1)), 5, (0,0,255), 1)
            """
            draw_params = dict(matchColor = (255,0,255), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = None, # draw only inliers
                   flags = 2)
            #cv2.drawMatches(self.Frames[frame_number], new_keypoints, self.Frames[frame_number-1], old_keypoints, goodMatches, None, **draw_params)
        return srcPoints, dstPoints
    @staticmethod
    def calculateTransformationMatrix(rotationMatrix, translationVector):
        print("calculating transformation matrix..")
        transformationMatrix = np.eye(4, dtype= np.float64)
        transformationMatrix[:3,:3] = rotationMatrix
        transformationMatrix[:3,3] = translationVector.T
        print("transformationMatrix =\n",transformationMatrix)
        return transformationMatrix

    def decomposeEssentialMatrix(self, EssentialMatrix, point_old, point_new):
        print("decomposing essential matrix")
        R1, R2, t = cv2.decomposeEssentialMat(EssentialMatrix)# https://docs.opencv.org/4.0.0/d9/d0c/group__calib3d.html#ga54a2f5b3f8aeaf6c76d4a31dece85d5d
        print("R1 = {}\nR2 = {}\nt = {}".format(R1, R2, t))
        T1 = self.calculateTransformationMatrix(R1, t)
        T2 = self.calculateTransformationMatrix(R2, t)
        T3 = self.calculateTransformationMatrix(R1, -t)
        T4 = self.calculateTransformationMatrix(R2, -t)

        T = [T1, T2, T3, T4]
        #print("T = [T1, T2, T3, T4] = \n{} ".format(T))
        # Make cameraMatrix homogeneous
        cameraMatrix = np.concatenate((self.CameraMatrix, np.zeros((3,1))), axis= 1) # make last column of the cameraMatrix 0 0 0
        print("cameraMatrix = \n",cameraMatrix) 
        possiblePoses = [cameraMatrix @ T1, cameraMatrix @ T2, cameraMatrix @ T3, cameraMatrix @ T4]
        print("possiblePoses1 = \n",possiblePoses[0])
        print("possiblePoses2 = \n ",possiblePoses[1].tolist())
        print("possiblePoses3 = \n",possiblePoses[2])
        print("possiblePoses4 = \n",possiblePoses[3])
        # We must take poses only in front of the camera
        pozitivePoses = []
        for P,T in zip(possiblePoses, T):
            
            print(P)
            points3D_old = cv2.triangulatePoints(cameraMatrix, P, point_old.T, point_new.T)
            points3D_new = T @ points3D_old
            # dehomogenize
            points3D_old = points3D_old[:3, : ]/points3D_old[3, : ]
            points3D_new = points3D_new[:3, : ]/points3D_new[3, : ]
            print("points3D_new: ",points3D_new)
            distance = sum(points3D_old[2, : ]>0) + sum(points3D_new[2, : ]>0)
            scale = np.mean(np.linalg.norm(points3D_old.T[:-1]-points3D_old.T[1:], axis = -1)/np.linalg.norm(points3D_new.T[:-1]-points3D_new.T[1:], axis = -1))
            pozitivePoses.append(distance + scale)
        print("pozitivePoses = ",pozitivePoses)
        max = np.argmax(pozitivePoses)
        if (max == 0):
            return R1, np.ndarray.flatten(t)
        elif(max == 1):
            return R2, np.ndarray.flatten(t)
        elif(max == 2):
            return R1, np.ndarray.flatten(-t)
        elif(max == 3):
            return R2, np.ndarray.flatten(-t)

    def estimatePose(self, oldPoints, newPoints):
        EssentialMatrix, mask = cv2.findEssentialMat(oldPoints, newPoints, self.CameraMatrix)# https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga0b166d41926a7793ab1c351dbaa9ffd4
        print("EssentialMatrix: ",EssentialMatrix)
        R, t = self.decomposeEssentialMatrix(EssentialMatrix, oldPoints, newPoints)
        print("R = {} t = {} ".format(R,t))
        return self.calculateTransformationMatrix(R, t)

def main():
    vo = VisualOdometryMono()
    
    prev_frame_time = 0
    new_frame_time = 0
    print("ola0")
    if vo.cam.isOpened():
        print("ola1")
        ret,frame = vo.cam.read()
        frame_number = 0
        vo.Frames.append(frame)
        output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'XVID'), 15.0, (vo.frame_width, vo.frame_height)) # 'M','J','P','G' #https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html
    else: 
        ret = False
    while ret :
        if len(vo.Frames) == 2:
            vo.Frames = []
            frame_number = -1
        ret,frame = vo.cam.read()
        #frame = cv2.imread("Seeker/TargetImages/arrow1.png")
        vo.Frames.append(frame)
        frame_number += 1
        frame = cv2.resize(frame,(vo.frame_width, vo.frame_height ))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # apply grayscale
        #frame =cv2.flip(frame,-1)

        #Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        
        points2D_old, points2D_new = vo.findMatches(frame_number, gray)
        transformationMatrix = vo.estimatePose(points2D_old, points2D_new)
        print("transformationMatrix = ",transformationMatrix)
        if(vo.visuals == 1):
            cv2.putText(frame,"FPS:{}".format(int(fps)),(15,15),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1,cv2.LINE_AA)# Displays fps
            cv2.imshow("realTimeCamera", frame)
        if(vo.record == 1):
            output.write(frame)   
        key=cv2.waitKey(1)
        if key==27:
            break
    cv2.destroyAllWindows()
    vo.cam.release()
    output.release()
# FEATURE DETECTION


# FEATURE MATCHING/TRACKING


# POSE ESTIMATION


# BUNDLE ADJUSTMENT


#


main()