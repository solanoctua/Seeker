import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.colors as mcol
import matplotlib.cm as cm
import matplotlib.animation as animation
numberOfFrames = 50



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


cameraMatrix =np.array( [ [1  , 0. , 1  ], 
                          [0. , 1  , 1  ],
                          [0. , 0. , 1. ] ])



cameraMatrix =np.array( [ [8.6297319278294219e+02,          0.          , 3.2410970150269952e+02], 
                 [           0.         ,8.6289007075149686e+02, 2.3151809145852621e+02],
                 [           0.         ,          0.          ,            1.         ] ])

distortionCoeffs =np.array( [ 1.3876171595109568e-01, -5.0495233579131149e-01, -1.4364550355797534e-03, 3.0938437583063767e-03, 1.3034844698951493e+00 ])

detector = cv2.ORB_create(nfeatures = 1000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=2)
search_params = {}
goodFeatureParams = dict( maxCorners = 500, qualityLevel = 0.01, minDistance = 3 ) # , blockSize = 5

flann = cv2.FlannBasedMatcher(index_params, search_params)
ratioThresh = 0.75


def computeKeyandDes(frame):
    features = cv2.goodFeaturesToTrack(np.mean(frame, axis=2).astype(np.uint8), mask = None, **goodFeatureParams)
    keypoints = [cv2.KeyPoint(x = feature[0][0], y = feature[0][1], _size = 20) for feature in features ]  # size: diameter of the meaningful keypoint neighborhood
    keypoints, descriptors = detector.compute(frame, keypoints)
    return keypoints,descriptors

def compute2DPoints(kp1, desc1, kp2, desc2, srcPoints, dstPoints):  # kp1 new, kp2 old 

    goodMatches = []
    points1 = []
    points2 = []
    idx1 = []
    idx2 = []
    matches = flann.knnMatch(desc1, desc2, k=2)
    x = 0
    for m,n in matches:
            if m.distance < ratioThresh * n.distance:   
                #print("keypoints[m.queryIdx], oldkeypoints[m.trainIdx]: ",keypoints[m.queryIdx], oldkeypoints[m.trainIdx])      
                if (kp1[m.queryIdx].pt  not in srcPoints or kp2[m.trainIdx].pt  not in dstPoints):
                    goodMatches.append([kp1[m.queryIdx], kp2[m.trainIdx]])  # Dmatch.querdIdx https://docs.opencv.org/4.5.2/d4/de0/classcv_1_1DMatch.html
                    
                    srcPoints.append(kp1[m.queryIdx].pt) # points in the new frame
                    dstPoints.append(kp2[m.trainIdx].pt) # points in the old frame
                    idx1.append(kp1[m.queryIdx].pt)
                    idx2.append(kp2[m.trainIdx].pt)
                else:
                    x+=1
    print(x," points already in the frame")
    idx1 = np.float32(idx1).reshape(-1,1,2)
    idx2 = np.float32(idx2).reshape(-1,1,2)
        
    M, inlinersMask = cv2.findHomography(idx1, idx2, cv2.RANSAC,5.0) # mask for inliners

    i = 0
    for (point1, point2) in goodMatches: # only inliner matches are good matches 
        if (inlinersMask[i] == 1):
            #points1.append(point1.pt)   #KeyPoint.pt  https://docs.opencv.org/4.5.2/d2/d29/classcv_1_1KeyPoint.html#ae6b87d798d3e181a472b08fa33883abe
            #points2.append(point2.pt)
            x1,y1 = point1.pt
            x2,y2 = point2.pt
            cv2.line(frame, (round(x1),round(y1)), (round(x2),round(y2)), (255,0,0),2)
            cv2.circle(frame, (int(x1), int(y1)), 5, (0,0,255), 1)
        i+=1

    points1 = idx1[inlinersMask==1]
    points2 = idx2[inlinersMask==1]

    points1 = np.float32(points1)
    points2 = np.float32(points2)

    return points1,points2


cap = cv2.VideoCapture("C:/Users/asus/Desktop/Samurai-Seeker/SLAM Test_1/Sample8.mp4")
#width
#cap.set(3,960)
#height
#cap.set(4,540)
if cap.isOpened():
    srcPoints = []
    dstPoints = []
    points1 = []
    points2 = []
    plotSize = 200
    Rpose = np.eye(3)
    Tpose = np.zeros((3,3))
    t = 1
    ret , frame = cap.read()
    frame = cv2.resize(frame,(2160//2, 4096//8))
    oldKeypoints,oldDescriptors = computeKeyandDes(frame)
else:
    ret = False

while ret:
    ret , frame = cap.read()
    frame = cv2.resize(frame,(2160//2, 4096//8))
    keypoints,descriptors = computeKeyandDes(frame)
    points1,points2 = compute2DPoints(keypoints, descriptors, oldKeypoints, oldDescriptors, srcPoints, dstPoints)

    
    
    if(points1.size > 0 and points2.size > 0):
        print("len(srcPoints) = ",len(srcPoints))
        # Find essential matrix to extract camera pose (rotation and translation matrices)
        #E , _= cv2.findEssentialMat(points1, points2, cameraMatrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        F , _ = cv2.findFundamentalMat(points1, points2,  method=cv2.RANSAC, ransacReprojThreshold = 3, confidence = 0.999)
        E = (cameraMatrix.T).dot(F).dot(cameraMatrix)

        points, rotationMatrix, translationMatrix, mask_pose = cv2.recoverPose(E, points2, points1,cameraMatrix)
        Rpose = Rpose.dot(rotationMatrix)
        translation_z = np.matrix([[0],[0],[float("{:.3f}".format(float(translationMatrix[2])))]])
        #Tpose = Tpose +(1*(translationMatrix.T).dot(Rpose))
        Tpose = Tpose + 0.5*((translation_z.T).dot(Rpose))
        z = np.matrix('0;0;1')  # 3x1 matrix
        pose = Tpose.T*z
        np.set_printoptions(suppress=True)
        # plot with matplotlib
        ax.scatter(pose[1],pose[0],pose[2],marker="x",color=cpick1.to_rgba(t),s = plotSize )



        # Optimize the points for triangulation
        """
        F , _ = cv2.findFundamentalMat(points1, points2,  method=cv2.RANSAC, ransacReprojThreshold = 3, confidence = 0.999)
        E_est = (cameraMatrix.T).dot(F).dot(cameraMatrix)
        print("E_est: ",E_est)
        """
        points1 = points1.reshape(1,-1,2)
        points2 = points2.reshape(1,-1,2)
        points1, points2 = cv2.correctMatches(F, points1, points2)
        #print("corrected points1:",points1)

        translation_mtx_T = np.transpose(translationMatrix)
        rotation_mtx_T = np.transpose(rotationMatrix)
        cam_position = -rotation_mtx_T * translationMatrix
        # Convert points to homogeneous
        """
        points1Homo = cv2.convertPointsToHomogeneous(points1.astype(np.float32))
        points2Homo = cv2.convertPointsToHomogeneous(points2.astype(np.float32))
        """
        points1Homo = cv2.convertPointsToHomogeneous(points1).T
        points2Homo = cv2.convertPointsToHomogeneous(points2).T
        
        # Projection matrices for triangulation
        projectionMatrix1 = np.zeros([3,4])
        projectionMatrix1[:3,:3] = np.eye(3)
        #Rt = np.hstack([rotation_mtx_T, -rotation_mtx_T.dot(translationMatrix)])
        Rt = np.concatenate((rotationMatrix, translationMatrix), 1)
        projectionMatrix2 = cameraMatrix.dot(Rt)

        # Triangulate
        points4D = cv2.triangulatePoints(projectionMatrix1, projectionMatrix2, points1.T, points2.T).T
        points4D = points4D[np.abs(points4D[:,3])>0.001]

        # convert from homogeneous coordinates to 3D
        points3D =  points4D[:, :3]/np.repeat( points4D[:, 3], 3).reshape(-1, 3)
        #points3D = points4D / points4D[3]
        #points3D = points4D[:, :3] / points4D[:, 3:4]
        points3D = points3D *100
        

        # plot with matplotlib
        
        Ys = points3D[:, 0]
        Zs = points3D[:, 1]
        Xs = points3D[:, 2]
        ax.scatter(Xs, Ys, Zs,marker ="1",color=cpick2.to_rgba(t))
        
        t+= 1
        print("*********************************************************")
        #print("cameraMatrix: ",cameraMatrix)
        print("Frame: ",t)
        print("Translation matrix-vector: ",translationMatrix)
        print("RotationMatrix: ",rotationMatrix)
        print("Essential Matrix: ", E)
        print("Fundamental Matrix: ", F)
        print("Pose:", pose)
        print("RT: ",Rt)
        print("Projection Matrix: ",projectionMatrix2)
        #print("Points3D: ",points3D)
        
        """
        _,rotation_vector, translation_vector = cv2.solvePnP(points3D,points1,cameraMatrix,distortionCoeffs,flags=cv2.SOLVEPNP_ITERATIVE)

        p2, _ = cv2.projectPoints(points3D, rotation_vector, -translation_vector, cameraMatrix, distortionCoeffs) # Reproject 3D points back into the 2D image plane
        p2 = np.float32(p2)
        #print(points1.size,len(p2))
        reprojection_error = 0
        for i in range (int(points1.size/2)):
            
            #print("ola[i]= ",ola[i])
            #print("p2[i]= ",p2[i][0])
            
            #reprojection_error =((points[i][0] - p2[i][0])**2+(points[i][1]-p2[i][1])**2)**(1/2)
            reprojection_error += cv2.norm(ola[i], p2[i][0], cv2.NORM_L2)
        print("reprojection_error: ",reprojection_error/int(points1.size/2))
        """
    oldKeypoints,oldDescriptors = keypoints,descriptors
    cv2.imshow("Sample", frame)
     
    if cv2.waitKey(1) == 27 or t == numberOfFrames:   #press esc to quit
        cv2.destroyAllWindows()
        cap.release()
        plt.colorbar(cpick1,label="Camera Pose Per Frame")
        plt.colorbar(cpick2,label="Point Cloud Per Frame")
    
        plt.savefig("C:/Users/asus/Desktop/Samurai-Seeker/SLAM Test_1/VSLAMmaster.png")
        plt.show()
        break
