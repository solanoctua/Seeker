import cv2, time
import numpy as np
import datetime

class CAM():
    def __init__(self, camera_id = 0 ):
        self.frame = None
        self.camera_id = camera_id
        self.cam = cv2.VideoCapture(self.camera_id)
        self.camera_matrix = np.array([[4.854063946505623335e+02,            0.          , 3.052856318255884958e+02],
                                [          0.            ,4.854079534951868027e+02, 2.390909613840788666e+02],
                                [          0.            ,            0.          , 1.                      ]])

        self.dist_coeffs = np.array([-1.347099963195809436e-01, 2.136884941468401855e-02, -7.346353886725681681e-02, 7.520179027764506419e-02])

        self.frame_width, self.frame_height = (640,640)  #1280,720
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.colorspace = "RGB"

        self.HSVmin_values = np.array([0, 0, 0]) #np.array([min_h, min_s, min_v])
        self.HSVmax_values = np.array([179, 255, 255]) #np.array([max_h, max_s, max_v])

    def HSVinput(self, HSVminmax):
        self.HSVmin_values = np.array([HSVminmax[0], HSVminmax[1], HSVminmax[2]])
        self.HSVmax_values = np.array([HSVminmax[3], HSVminmax[4], HSVminmax[5]])

    def startFeed(self):
        if self.cam.isOpened():
            ret,self.frame = self.cam.read()
            if self.colorspace == "HSV":
                self.frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
                #print(self.HSVmin_values)
                #print(self.HSVmax_values)
                mask_HSV = cv2.inRange(self.frame, self.HSVmin_values, self.HSVmax_values) 
                target = cv2.bitwise_and(self.frame, self.frame, mask = mask_HSV)
                return target
            if ret:
                pass
            
            return self.frame
        else:
            print("CAM%d is failed to open"% self.camera_id)
        
    def stopFeed(self):
        if self.cam.isOpened():
            self.cam.release()

    def saveFrame(self, path = "C:/Users/asus/Desktop"):
        date = datetime.datetime.now()

        print("image saved to {}".format(path))
        cv2.imwrite("{}/{}.{}.{}.{}_{}.jpg".format(path, date.day, date.month, date.year, date.minute, date.second),self.frame )



if __name__ == "__main__":
    cam = CAM()
    cam.colorspace = "HSV"
    t = 0
    prev_frame_time = 0
    new_frame_time = 0
    while True:
        frame = cam.startFeed()      
        
        #Calculate FPS
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(frame,"FPS:{}".format(int(fps)),(15,25),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,0,0),2,cv2.LINE_AA)# Displays fps
        cv2.imshow("Real Time Frame", frame)
        if t == 10:
            pass
            #cam.saveFrame()
        t+= 1
        key=cv2.waitKey(1)
        if key==27:
            break
    cv2.destroyAllWindows()
    cam.stopFeed()