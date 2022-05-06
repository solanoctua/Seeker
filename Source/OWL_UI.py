from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QColor
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import cv2
import time, datetime, os
from CAM import *
class Worker_CAM(QThread):
    frameUpdate = pyqtSignal(QImage, int)
    def __init__(self,camera_id ):
        self.camera_id = camera_id
        self.cam = CAM(self.camera_id)
        QtCore.QThread.__init__(self)
        
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv_img
        #rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(530,530, Qt.KeepAspectRatio)
        self.frameUpdate.emit(p,self.camera_id)

    def run(self): # when Worker_CAM.start() is called, this function will run
        self.ThreadActive = True
        prev_frame_time = 0
        new_frame_time = 0
        while self.ThreadActive:
            frame = cv2.flip(self.cam.startFeed(),1)
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                #print("CAM%d is not working properly"%self.camera_id )
                break
            #Calculate FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            cv2.putText(frame,"FPS:{}".format(int(fps)),(15,25),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2,cv2.LINE_AA)# Displays fps
            
            qt_img = self.convert_cv_qt(frame)

        """
        cam = cv2.VideoCapture(self.camera_id)
    
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 360)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 620)
        prev_frame_time = 0
        new_frame_time = 0
        while self.ThreadActive:
            if cam.isOpened():
                ret, frame = cam.read()
                frame=cv2.flip(frame,1)
                if ret:
                    #Calculate FPS
                    new_frame_time = time.time()
                    fps = 1/(new_frame_time-prev_frame_time)
                    prev_frame_time = new_frame_time
                    cv2.putText(frame,"FPS:{}".format(int(fps)),(15,15),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1,cv2.LINE_AA)# Displays fps
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    qt_img = self.convert_cv_qt(frame)
                    
                    #self.CAM0_label.update()
                    """
            
    def stop(self):
        self.cam.stopFeed()
        self.ThreadActive = False
        self.quit()

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.isCAM0Running = 0
        self.isCAM1Running = 0
        self.isCAM2Running = 0
        currentWorkingDirectory = os.getcwd()
        self.savePATH = currentWorkingDirectory.replace("\\", "/")

        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1920, 1000)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Create GroupBox for CAM Labels
        self.groupBox_CAMS = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_CAMS.setGeometry(QtCore.QRect(20, 120, 1881, 481))
        self.groupBox_CAMS.setObjectName("groupBox_CAMS")
        # Create Widget for CAM Labels
        self.widget = QtWidgets.QWidget(self.groupBox_CAMS)
        self.widget.setGeometry(QtCore.QRect(20, 30, 1841, 401))
        self.widget.setObjectName("widget")
        # Create CAM Labels
        self.label_CAM0 = QtWidgets.QLabel(self.widget)
        self.label_CAM0.setObjectName("label_CAM0")
        self.label_CAM1 = QtWidgets.QLabel(self.widget)
        self.label_CAM1.setObjectName("label_CAM1")
        self.label_CAM2 = QtWidgets.QLabel(self.widget)
        self.label_CAM2.setObjectName("label_CAM2")
        # Horizontal Layout for CAM Labels
        self.horizontalLayout_CAMS = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_CAMS.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_CAMS.setObjectName("horizontalLayout")
        self.horizontalLayout_CAMS.addWidget(self.label_CAM0)
        self.horizontalLayout_CAMS.addWidget(self.label_CAM1)
        self.horizontalLayout_CAMS.addWidget(self.label_CAM2)

        # Create Widget for Take Picture Buttons
        self.widget1 = QtWidgets.QWidget(self.groupBox_CAMS)
        self.widget1.setGeometry(QtCore.QRect(20, 440, 1841, 30))
        self.widget1.setObjectName("widget1")
        # Create Take Picture Buttons for each Camera Feed
        self.pushButton_TakePic_CAM0 = QtWidgets.QPushButton(self.widget1)
        self.pushButton_TakePic_CAM0.setObjectName("pushButton_TakePic_CAM0")
        self.pushButton_TakePic_CAM1 = QtWidgets.QPushButton(self.widget1)
        self.pushButton_TakePic_CAM1.setObjectName("pushButton_TakePic_CAM1")
        self.pushButton_TakePic_CAM2 = QtWidgets.QPushButton(self.widget1)
        self.pushButton_TakePic_CAM2.setObjectName("pushButton_TakePic_CAM2")

        self.pushButton_TakePic_CAM0.clicked.connect(lambda: self.takePic(camera_id = 0))
        self.pushButton_TakePic_CAM1.clicked.connect(lambda: self.takePic(camera_id = 1))
        self.pushButton_TakePic_CAM2.clicked.connect(lambda: self.takePic(camera_id = 2))
        # Horizontal Layout for Take Picture Buttons
        self.horizontalLayout_TakePicButtons = QtWidgets.QHBoxLayout(self.widget1)
        self.horizontalLayout_TakePicButtons.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_TakePicButtons.setObjectName("horizontalLayout_2")
        self.horizontalLayout_TakePicButtons.addWidget(self.pushButton_TakePic_CAM0)
        self.horizontalLayout_TakePicButtons.addWidget(self.pushButton_TakePic_CAM1)
        self.horizontalLayout_TakePicButtons.addWidget(self.pushButton_TakePic_CAM2)

        # Create GroupBox for Connect Section
        self.groupBox_Connect = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_Connect.setGeometry(QtCore.QRect(1620, 10, 281, 111))
        self.groupBox_Connect.setObjectName("groupBox_Connect")
        # Create Connect Button
        self.pushButton_Connect = QtWidgets.QPushButton(self.groupBox_Connect)
        self.pushButton_Connect.setGeometry(QtCore.QRect(140, 30, 120, 40))
        self.pushButton_Connect.setObjectName("pushButton_Connect")
        self.pushButton_Connect.setStyleSheet("background-color: green;")
        self.pushButton_Connect.clicked.connect(lambda: self.changeButton(self.pushButton_Connect))
        # Create GroupBox for Something in future
        self.groupBox_PlaceHolder = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_PlaceHolder.setGeometry(QtCore.QRect(20, 10, 1601, 111))
        self.groupBox_PlaceHolder.setObjectName("groupBox_PlaceHolder")


        # Create Widget for STATUS and COMMAND Sections
        self.widget2 = QtWidgets.QWidget(self.centralwidget)
        self.widget2.setGeometry(QtCore.QRect(20, 610, 1881, 311))
        self.widget2.setObjectName("widget2")

    
        # Create GroupBox for ***STATUS*** Section 
        self.groupBox_Status = QtWidgets.QGroupBox(self.widget2)
        self.groupBox_Status.setObjectName("groupBox_Status")
        # Create TextBrowser 
        self.textBrowser_Status = QtWidgets.QTextBrowser(self.groupBox_Status)
        self.textBrowser_Status.setGeometry(QtCore.QRect(10, 20, 651, 241)) #  QtCore.QRect(20, 20, 631, 261)
        self.textBrowser_Status.setObjectName("textBrowser_Status")
        # Create Save Button to Save Data in the TextBrowser
        self.pushButton_Status_Save = QtWidgets.QPushButton(self.groupBox_Status)
        self.pushButton_Status_Save.setGeometry(QtCore.QRect(540, 270, 93, 31)) #
        self.pushButton_Status_Save.setObjectName("pushButton_Status_Save")
        self.pushButton_Status_Save.clicked.connect(self.saveStatus)
        # Create Clear Button to Save Data in the TextBrowser
        self.pushButton_Status_Clear = QtWidgets.QPushButton(self.groupBox_Status)
        self.pushButton_Status_Clear.setObjectName("pushButton_Status_Clear")
        self.pushButton_Status_Clear.setGeometry(QtCore.QRect(440, 270, 93, 31))
        self.pushButton_Status_Clear.clicked.connect(self.clearStatusBar)
        # Create GroupBox for GPS Related Data: Coordinates, VehicleSpeed, VehicleAltitude
        self.groupBox = QtWidgets.QGroupBox(self.groupBox_Status)
        self.groupBox.setGeometry(QtCore.QRect(670, 20, 271, 131))
        self.groupBox.setObjectName("groupBox")
        # Create TextBrowser to Display Coordinates, VehicleSpeed and VehicleAltitude
        self.textBrowser_Coords = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_Coords.setGeometry(QtCore.QRect(10, 20, 251, 41))
        self.textBrowser_Coords.setObjectName("textBrowser_Coords")
        self.textBrowser_Speed = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_Speed.setGeometry(QtCore.QRect(10, 70, 121, 41))
        self.textBrowser_Speed.setObjectName("textBrowser_Speed")
        self.textBrowser_Altitude = QtWidgets.QTextBrowser(self.groupBox)
        self.textBrowser_Altitude.setGeometry(QtCore.QRect(140, 70, 121, 41))
        self.textBrowser_Altitude.setObjectName("textBrowser_Altitude")
        # Create GroupBox for Battery Related Data: 
        self.groupBox_2 = QtWidgets.QGroupBox(self.groupBox_Status)
        self.groupBox_2.setGeometry(QtCore.QRect(670, 150, 271, 141))
        self.groupBox_2.setObjectName("groupBox_2")
        # Create TextBrowser to Display TotalVoltage, CellVoltage, AmpDrawn and MahUsed
        self.textBrowser_TotalVoltage = QtWidgets.QTextBrowser(self.groupBox_2)
        self.textBrowser_TotalVoltage.setGeometry(QtCore.QRect(10, 30, 121, 41))
        self.textBrowser_TotalVoltage.setObjectName("textBrowser_TotalVoltage")
        self.textBrowser_AmpDraw = QtWidgets.QTextBrowser(self.groupBox_2)
        self.textBrowser_AmpDraw.setGeometry(QtCore.QRect(140, 30, 121, 41))
        self.textBrowser_AmpDraw.setObjectName("textBrowser_AmpDraw")
        self.textBrowser_CellVoltage = QtWidgets.QTextBrowser(self.groupBox_2)
        self.textBrowser_CellVoltage.setGeometry(QtCore.QRect(10, 80, 121, 41))
        self.textBrowser_CellVoltage.setObjectName("textBrowser_CellVoltage")
        self.textBrowser_MahUsed = QtWidgets.QTextBrowser(self.groupBox_2)
        self.textBrowser_MahUsed.setGeometry(QtCore.QRect(140, 80, 121, 41))
        self.textBrowser_MahUsed.setObjectName("textBrowser_MahUsed")

        
        # Create GroupBox for ***COMMANDS*** Section
        self.groupBox_Commands = QtWidgets.QGroupBox(self.widget2)
        self.groupBox_Commands.setObjectName("groupBox_Commands")
        # Create Servo Slider for Water Sampling Mechanism
        self.verticalSlider_Servo0 = QtWidgets.QSlider(self.groupBox_Commands)
        self.verticalSlider_Servo0.setGeometry(QtCore.QRect(880, 30, 20, 251))
        self.verticalSlider_Servo0.setOrientation(QtCore.Qt.Vertical)
        self.verticalSlider_Servo0.setObjectName("verticalSlider_Servo0")
        # Create Widget for CommandButtons
        self.widget3 = QtWidgets.QWidget(self.groupBox_Commands)
        self.widget3.setGeometry(QtCore.QRect(10, 20, 279, 137))
        self.widget3.setObjectName("widget3")
        # Create Basic Command Buttons: ARM, LAND, RTH(Return to Home) and POSHOLD(Position Hold)
        self.pushButton_ARM = QtWidgets.QPushButton(self.widget3)
        self.pushButton_ARM.setObjectName("pushButton_ARM")
        self.pushButton_ARM.setEnabled(False)
        self.pushButton_ARM.clicked.connect(lambda: self.changeButton(self.pushButton_ARM))

        self.pushButton_LAND = QtWidgets.QPushButton(self.widget3)
        self.pushButton_LAND.setObjectName("pushButton_LAND")
        self.pushButton_LAND.setEnabled(False)
        self.pushButton_RTH = QtWidgets.QPushButton(self.widget3)
        self.pushButton_RTH.setObjectName("pushButton_RTH")
        self.pushButton_RTH.setEnabled(False)
        self.pushButton_POSHOLD = QtWidgets.QPushButton(self.widget3)
        self.pushButton_POSHOLD.setObjectName("pushButton_POSHOLD")
        self.pushButton_POSHOLD.setEnabled(False)
        # Create Vertical Layout for Basic Commands Buttons
        self.verticalLayout_BasicCommands = QtWidgets.QVBoxLayout()
        self.verticalLayout_BasicCommands.setObjectName("verticalLayout_BasicCommands")
        self.verticalLayout_BasicCommands.addWidget(self.pushButton_ARM)
        self.verticalLayout_BasicCommands.addWidget(self.pushButton_LAND)
        self.verticalLayout_BasicCommands.addWidget(self.pushButton_RTH)
        self.verticalLayout_BasicCommands.addWidget(self.pushButton_POSHOLD)
        # Create Mission Command Buttons: START MISSION, ABORT MISSION, 1 and 2
        self.pushButton_ABORT = QtWidgets.QPushButton(self.widget3)
        self.pushButton_ABORT.setObjectName("pushButton_ABORT")
        self.pushButton_START_MISSION = QtWidgets.QPushButton(self.widget3)
        self.pushButton_START_MISSION.setObjectName("pushButton_START_MISSION")
        self.pushButton_1 = QtWidgets.QPushButton(self.widget3)
        self.pushButton_1.setObjectName("pushButton_1")
        self.pushButton_ARM_10 = QtWidgets.QPushButton(self.widget3)
        self.pushButton_ARM_10.setObjectName("pushButton_ARM_10")
        # Create Vertical Layout for Mission Commands Buttons
        self.verticalLayout_MissionCommands = QtWidgets.QVBoxLayout()
        self.verticalLayout_MissionCommands.setObjectName("verticalLayout_MissionCommands")
        self.verticalLayout_MissionCommands.addWidget(self.pushButton_ABORT)
        self.verticalLayout_MissionCommands.addWidget(self.pushButton_START_MISSION)
        self.verticalLayout_MissionCommands.addWidget(self.pushButton_1)
        self.verticalLayout_MissionCommands.addWidget(self.pushButton_ARM_10)
        # Create Camera CheckBoxes
        self.checkBox_CAM0 = QtWidgets.QCheckBox(self.widget3)
        self.checkBox_CAM0.setObjectName("checkBox_CAM0")
        self.checkBox_CAM1 = QtWidgets.QCheckBox(self.widget3)
        self.checkBox_CAM1.setObjectName("checkBox_CAM1")
        self.checkBox_CAM2 = QtWidgets.QCheckBox(self.widget3)
        self.checkBox_CAM2.setObjectName("checkBox_CAM2")
        self.checkBox_CAM0.stateChanged.connect(lambda: self.openCam(self.checkBox_CAM0))
        self.checkBox_CAM1.stateChanged.connect(lambda: self.openCam(self.checkBox_CAM1))
        self.checkBox_CAM2.stateChanged.connect(lambda: self.openCam(self.checkBox_CAM2))
        # Create Vertical Layout for Camera CheckBoxes
        self.verticalLayout_CAMCheckBoxes = QtWidgets.QVBoxLayout()
        self.verticalLayout_CAMCheckBoxes.setObjectName("verticalLayout_CAMCheckBoxes")
        self.verticalLayout_CAMCheckBoxes.addWidget(self.checkBox_CAM0)
        self.verticalLayout_CAMCheckBoxes.addWidget(self.checkBox_CAM1)
        self.verticalLayout_CAMCheckBoxes.addWidget(self.checkBox_CAM2)
        # Create Horizontal Layout for Basic Buttons, Mission Buttons and Camera CheckBoxes in COMMANDS Section
        self.horizontalLayout_CommandButtons = QtWidgets.QHBoxLayout(self.widget3)
        self.horizontalLayout_CommandButtons.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_CommandButtons.setObjectName("horizontalLayout_CommandButtons")
        self.horizontalLayout_CommandButtons.addLayout(self.verticalLayout_BasicCommands)
        self.horizontalLayout_CommandButtons.addLayout(self.verticalLayout_MissionCommands)
        self.horizontalLayout_CommandButtons.addLayout(self.verticalLayout_CAMCheckBoxes)

        # Horizontal Layout for STATUS and COMMANDS Sections
        self.horizontalLayout_StatusAndCommands = QtWidgets.QHBoxLayout(self.widget2)
        self.horizontalLayout_StatusAndCommands.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_StatusAndCommands.setObjectName("horizontalLayout_StatusAndCommands")
        self.horizontalLayout_StatusAndCommands.addWidget(self.groupBox_Status)
        self.horizontalLayout_StatusAndCommands.addWidget(self.groupBox_Commands)


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1920, 26))
        self.menubar.setObjectName("menubar")

        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuSettings = QtWidgets.QMenu(self.menubar)
        self.menuSettings.setObjectName("menuSettings")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.actionScreenShot = QtWidgets.QAction(MainWindow)
        self.actionScreenShot.setObjectName("actionScreenShot")
        self.menuFile.addAction(self.actionScreenShot)

        self.PathforSave = QtWidgets.QAction(MainWindow)
        self.PathforSave.setObjectName("PathforSave")
        self.menuSettings.addAction(self.PathforSave)
        self.PathforSave.triggered.connect(self.savePath)
        
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuSettings.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "OWL Management"))
        self.groupBox_CAMS.setTitle(_translate("MainWindow", "CAMERA FEEDS"))
        self.label_CAM0.setText(_translate("MainWindow", "CAM0_FEED"))
        self.label_CAM1.setText(_translate("MainWindow", "CAM1_FEED"))
        self.label_CAM2.setText(_translate("MainWindow", "CAM2_FEED"))
        self.pushButton_TakePic_CAM0.setText(_translate("MainWindow", "TAKE PIC"))
        self.pushButton_TakePic_CAM1.setText(_translate("MainWindow", "TAKE PIC"))
        self.pushButton_TakePic_CAM2.setText(_translate("MainWindow", "TAKE PIC"))
        self.groupBox_Connect.setTitle(_translate("MainWindow", "CONNECT"))
        self.pushButton_Connect.setText(_translate("MainWindow", "CONNECT"))
        self.groupBox_PlaceHolder.setTitle(_translate("MainWindow", "PLACEHOLDER"))
        self.groupBox_Status.setTitle(_translate("MainWindow", "STATUS"))
        self.pushButton_Status_Save.setText(_translate("MainWindow", "SAVE"))
        self.pushButton_Status_Clear.setText(_translate("MainWindow", "CLEAR"))
        self.groupBox.setTitle(_translate("MainWindow", "GPS"))
        self.groupBox_2.setTitle(_translate("MainWindow", "BATTERY"))
        self.groupBox_Commands.setTitle(_translate("MainWindow", "COMMANDS"))
        self.pushButton_ARM.setText(_translate("MainWindow", "ARM"))
        self.pushButton_LAND.setText(_translate("MainWindow", "LAND"))
        self.pushButton_RTH.setText(_translate("MainWindow", "RTH"))
        self.pushButton_POSHOLD.setText(_translate("MainWindow", "POSHOLD"))
        self.pushButton_ABORT.setText(_translate("MainWindow", "MISSION ABORT"))
        self.pushButton_START_MISSION.setText(_translate("MainWindow", "START MISSION"))
        self.pushButton_1.setText(_translate("MainWindow", "1"))
        self.pushButton_ARM_10.setText(_translate("MainWindow", "2"))
        self.checkBox_CAM0.setText(_translate("MainWindow", "CAM0"))
        self.checkBox_CAM1.setText(_translate("MainWindow", "CAM1"))
        self.checkBox_CAM2.setText(_translate("MainWindow", "CAM2"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuSettings.setTitle(_translate("MainWindow", "Settings"))
        self.menuHelp.setTitle(_translate("MainWindow", "Help"))
        self.actionScreenShot.setText(_translate("MainWindow", "ScreenShot"))
        self.PathforSave.setText(_translate("MainWindow", "PathforSave"))

    def printIntoTextbox(self, message):
        redColor = QColor(255, 0, 0)
        greenColor = QColor(0, 255, 0)
        blueColor = QColor(0, 0, 255)
        blackColor = QColor(0, 0, 0)

        self.textBrowser_Status.setTextColor(redColor)
        self.textBrowser_Status.append(message)
    
    def savePath(self):
        
        dialog = QtWidgets.QFileDialog()
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        dialog.setWindowTitle("Path to save pictures, status raport, etc.")
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptOpen)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        if dialog.exec_() == QtWidgets.QFileDialog.Accepted:
            #print(dialog.selectedFiles()[0]) 
            self.printIntoTextbox("Save path is changed to %s" % dialog.selectedFiles()[0].replace("/", "\\"))
            self.savePATH = dialog.selectedFiles()[0]

    def clearStatusBar(self):
        self.textBrowser_Status.clear()     

    def saveStatus(self):
        date = datetime.datetime.now()
        with open("{}/StatusBar_{}.{}.{}_{}-{}.txt".format(self.savePATH, date.day, date.month, date.year, date.minute, date.second), 'w') as textfile:
            textfile.write(self.textBrowser_Status.toPlainText())
        self.printIntoTextbox("Status saved.")

    def takePic(self, camera_id):
        if camera_id == 0:
            if self.isCAM0Running:

                self.Worker_CAM0.cam.saveFrame(self.savePATH)
                self.printIntoTextbox("Picture is saved.")
            else:
                self.printIntoTextbox("CAM0 is not open, picture cannot be taken!")
        if camera_id == 1:
            if self.isCAM1Running:

                self.Worker_CAM1.cam.saveFrame(self.savePATH)
                self.printIntoTextbox("Picture is saved.")
            else:
                self.printIntoTextbox("CAM1 is not open, picture cannot be taken!")
        if camera_id == 2:
            if self.isCAM2Running:

                self.Worker_CAM2.cam.saveFrame(self.savePATH)
                self.printIntoTextbox("Picture is saved.")
            else:
                self.printIntoTextbox("CAM2 is not open, picture cannot be taken!")

    def changeButton(self, button):
        print(button.text())
        if button.text() == "CONNECT":
            self.printIntoTextbox("Connecting to OWL..")
            self.pushButton_ARM.setEnabled(True)
            self.pushButton_LAND.setEnabled(True)
            self.pushButton_RTH.setEnabled(True)
            self.pushButton_POSHOLD.setEnabled(True)

            self.pushButton_ARM.setStyleSheet("background-color: green;")
            button.setText("DISCONNECT")
            button.setStyleSheet("background-color: red;")
            
            return 1
        if button.text() == "DISCONNECT":
            self.printIntoTextbox("Disconnecting..")
            self.pushButton_ARM.setEnabled(False)
            self.pushButton_LAND.setEnabled(False)
            self.pushButton_RTH.setEnabled(False)
            self.pushButton_POSHOLD.setEnabled(False)

            self.pushButton_ARM.setStyleSheet("background-color: none;")
            button.setText("CONNECT")
            button.setStyleSheet("background-color: green;") 
            return 1
        if button.text() == "ARM":
            self.printIntoTextbox("Arming motors..")
            button.setText("DISARM")
            button.setStyleSheet("background-color: red;") 
            return 1
        if button.text() == "DISARM":
            self.printIntoTextbox("Disarming motors..")
            button.setText("ARM")
            button.setStyleSheet("background-color: green;") 
            return 1
    def updateFrame(self, qtframe, camera_id):
        """Updates the image_label with a new opencv image"""
        if camera_id == 0:
            label = self.label_CAM0
        if camera_id == 1:
            label = self.label_CAM1
        if camera_id == 2:
            label = self.label_CAM2
        label.setPixmap(QPixmap.fromImage(qtframe)) 

    def takeSliderInput(self):
        pass
    def openCam(self,checkbox):
        device = checkbox.text()
    
        if checkbox.isChecked(): #https://linuxhint.com/use-pyqt-checkbox/
            self.printIntoTextbox("{} is opened.".format(device))
            if device == "CAM0":
                self.isCAM0Running = 1
                self.Worker_CAM0 = Worker_CAM(0)
                self.Worker_CAM0.start()
                self.Worker_CAM0.frameUpdate.connect(self.updateFrame)  #lambda: self.updateFrame(label = self.label_CAM0)
            if device == "CAM1":
                self.isCAM1Running = 1
                self.Worker_CAM1 = Worker_CAM(1)
                try:
                    self.Worker_CAM1.start()
                except:
                    self.printIntoTextbox("{} is not found".format(device))
                    return 0
                self.Worker_CAM1.frameUpdate.connect(self.updateFrame)
            if device == "CAM2":
                self.isCAM2Running = 1
                self.Worker_CAM2 = Worker_CAM(2)
                try:
                    self.Worker_CAM2.start()
                except:
                    self.printIntoTextbox("{} is not found".format(device))
                    return 0
                self.Worker_CAM2.frameUpdate.connect(self.updateFrame)

        else:
            if device == "CAM0":
                self.isCAM0Running = 0
                self.printIntoTextbox("{} is closed.".format(device))
                self.Worker_CAM0.stop()
            if device == "CAM1":
                self.isCAM0Running = 0
                self.printIntoTextbox("{} is closed.".format(device))
                self.Worker_CAM1.stop()
            if device == "CAM2":
                self.isCAM0Running = 0
                self.printIntoTextbox("{} is closed.".format(device))
                self.Worker_CAM2.stop()

    
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    ui.printIntoTextbox("Current save path is %s" % ui.savePATH.replace( "/","\\"))
    sys.exit(app.exec_())
    # when exit QUAD SHOULD BEHAVE ?
