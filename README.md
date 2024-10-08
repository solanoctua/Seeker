# Seeker

This repository contains prototype algorithms for quadcopter which executes autonomous missions.
Precise landing on Aruco markers, depth estimation and obstacle avoidance with stereo cameras.

Libraries: OpenCV(version 4.0), Dronekit, Numpy, Matplotlib(for 2D-3D Plotting), Skimage, Pytesseract(for text reading)

| Quadcopter Specifications  |  | 
| :---         | :---           | 
| Frame:   | 7 inch Hybrid-X configuration   | 
| Running Firmware:     | https://firmware.ardupilot.org/Copter/latest/KakuteF7/ (will be upgrated to INAV 4.0)    | 
| Flight Controller:   | Holybro Kakute F7 (http://www.holybro.com/product/kakute-f7-v1-5/)  | 
| Companion Computer:    | Raspberry Pi 4 Compute Module + StereoPi V2 (https://www.stereopi.com/v2)     | 
| Communication:  | Over WIFI with ESP32-M1-Reach-Out  (https://www.crowdsupply.com/bison-science/esp32-m1-reach-out)    | 
| GPS: | http://www.mateksys.com/?portfolio=m9n-f4 |
| ESC:     | Tekko32 F3 45A 4 In 1 Blheli 32 3-6S Brushless ESC      | 
| Motors:  |  iflight XING 2450KV Brushless Motor x4 (will be upgraded to https://shop.iflight-rc.com/quad-parts-cat20/motors-cat26/xing-motors-cat148/xing-x2806-5-fpv-nextgen-motor-pro1001)     | 
| Cameras:     | -For landing: https://www.waveshare.com/imx335-5mp-usb-camera-a.htm (auto focus version recommended)      -For object avoidance and mapping: https://www.waveshare.com/imx219-160-camera.htm x2       | 

FPV components are used for testing, troubleshooting and safety purposes only.

[![Build](https://github.com/solanoctua/Seeker/blob/main/Stuff/Seeker.jpg)](https://youtu.be/mLf-d8wXq1Y)
[![Camera Calibration](https://img.youtube.com/vi/YAxB-z1O-gI/0.jpg)](https://youtu.be/YAxB-z1O-gI)
[![Camera Calibration Results](https://img.youtube.com/vi/003jSb1dTzg/0.jpg)](https://youtu.be/003jSb1dTzg)


![ArucoMarkerDetection](https://github.com/solanoctua/Seeker/blob/main/Stuff/ArucoLock.png?raw=true)
<p float="left">
<img src="https://github.com/solanoctua/Seeker/blob/main/Stuff/ArrowDirection1.png" width="270" height="270">
<img src="https://github.com/solanoctua/Seeker/blob/main/Stuff/ArrowDirection2.png" width="270" height="270">
<img src="https://github.com/solanoctua/Seeker/blob/main/Stuff/ArrowDirection3.png" width="270" height="270">
</p>

![DepthCalibration](https://github.com/solanoctua/Seeker/blob/main/Stuff/depth_calib_correct.png?raw=true)
