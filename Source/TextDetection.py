import cv2, time, math
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

def nothing(x):
    pass

cam = cv2.VideoCapture(0)
if cam.isOpened():
    ret,frame = cam.read()
    frame_width, frame_height = (640,640)
    #output = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 20, (frame_width, frame_height)) #https://docs.opencv.org/3.4/dd/d9e/classcv_1_1VideoWriter.html
else: 
    ret = False
while ret :
    ret,frame = cam.read()
    frame = cv2.imread("Seeker/TargetImages/arrow1.png")
    frame = cv2.resize(frame,(frame_width, frame_height ))
    # convert to HSV colorspace 
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    min_color = np.array([0, 0, 0])
    max_color = np.array([179, 255, 100])
    mask_color = cv2.inRange(hsv_frame, min_color, max_color)
    # Dilate to connect text characters
    kernel = np.ones((3, 3), 'uint8')
    mask_color = cv2.dilate(mask_color, kernel, iterations =4) 
    # Find all text as a one contour
    contours, hierarchy = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #SIMPLE-NONE
    contours = sorted(contours, key = cv2.contourArea)
    target_contours = contours[-2:-1] # Take the object with the largest area
    for contour in target_contours:
        if cv2.contourArea(contour) >= 500: # If area is big enough, find its center etc.
            # Find smallest rectangle that encloses the text
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
            tolerance = 10
            if min_height - tolerance >= 0:
                min_height -= tolerance
            if max_height + tolerance <= frame_height:
                max_height += tolerance
            if min_width - tolerance >= 0:
                min_width -= tolerance
            if max_width + tolerance <= frame_width:
                max_width += tolerance
            
            # Cropping text area as an input to OCR
            text_area = frame[ min_width : max_width , min_height : max_height]
            # Read the text https://muthu.co/all-tesseract-ocr-options/
            #print("text: ",pytesseract.image_to_string(text)) #, config='digits'
            text = pytesseract.image_to_string(text_area, lang='eng',config='--psm 6') #--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789
            text = pytesseract.image_to_string(text_area, lang='eng',config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
            cv2.drawContours(frame,[ROI],0,(0,0,255),2)
            text = text.replace(" ", "")
            print(text[:3])
            print(len(text[:3]))
            if text[:3].isdigit():
                print("found",text[:3]) 
   
    cv2.imshow("mask", mask_color)
    cv2.imshow("text", text_area) 
    cv2.imshow("results", frame)
    
    #output.write(frame)  
    
    key=cv2.waitKey(1)
    if key==27:
        break
cv2.destroyAllWindows()
#output.release()
cam.release()