# -*- coding: utf-8 -*-
from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_PopWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("HSV Input")
        MainWindow.resize(600, 340)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_HSVColorSpace = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_HSVColorSpace.setGeometry(QtCore.QRect(10, 9, 581, 311))
        self.groupBox_HSVColorSpace.setObjectName("groupBox_HSVColorSpace")
        self.pushButton_SaveHSV = QtWidgets.QPushButton(self.groupBox_HSVColorSpace)
        self.pushButton_SaveHSV.setGeometry(QtCore.QRect(470, 270, 93, 28))
        self.pushButton_SaveHSV.setObjectName("pushButton_SaveHSV")

        

        self.widget1 = QtWidgets.QWidget(self.groupBox_HSVColorSpace)
        self.widget1.setGeometry(QtCore.QRect(110, 10, 461, 271))
        self.widget1.setObjectName("widget1")


        self.verticalLayout_Sliders = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout_Sliders.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_Sliders.setObjectName("verticalLayout_Sliders")

        self.verticalLayout_Slidersmin = QtWidgets.QVBoxLayout()
        self.verticalLayout_Slidersmin.setObjectName("verticalLayout_Slidersmin")



        self.horizontalSlider_Hmin = QtWidgets.QSlider(self.widget1)
        self.horizontalSlider_Hmin.setMaximum(179)
        self.horizontalSlider_Hmin.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_Hmin.setObjectName("horizontalSlider_Hmin")
        self.horizontalSlider_Hmin.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.horizontalSlider_Hmin.setTickInterval(10)
        
        self.horizontalSlider_Smin = QtWidgets.QSlider(self.widget1)
        self.horizontalSlider_Smin.setMaximum(255)
        self.horizontalSlider_Smin.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_Smin.setObjectName("horizontalSlider_Smin")
        self.horizontalSlider_Smin.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.horizontalSlider_Smin.setTickInterval(10)
        
        self.horizontalSlider_Vmin = QtWidgets.QSlider(self.widget1)
        self.horizontalSlider_Vmin.setMaximum(255)
        self.horizontalSlider_Vmin.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_Vmin.setObjectName("horizontalSlider_Vmin")
        self.horizontalSlider_Vmin.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.horizontalSlider_Vmin.setTickInterval(10)

        self.horizontalSlider_Hmin.valueChanged[int].connect(lambda: self.takeSliderInput(self.horizontalSlider_Hmin))
        self.horizontalSlider_Smin.valueChanged[int].connect(lambda: self.takeSliderInput(self.horizontalSlider_Smin))
        self.horizontalSlider_Vmin.valueChanged[int].connect(lambda: self.takeSliderInput(self.horizontalSlider_Vmin))

        self.verticalLayout_Slidersmin.addWidget(self.horizontalSlider_Hmin)
        self.verticalLayout_Slidersmin.addWidget(self.horizontalSlider_Smin)
        self.verticalLayout_Slidersmin.addWidget(self.horizontalSlider_Vmin)

        self.verticalLayout_Slidersmax = QtWidgets.QVBoxLayout()
        self.verticalLayout_Slidersmax.setObjectName("verticalLayout_Slidersmax")

        self.horizontalSlider_Hmax = QtWidgets.QSlider(self.widget1)
        self.horizontalSlider_Hmax.setMaximum(179)
        self.horizontalSlider_Hmax.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_Hmax.setObjectName("horizontalSlider_Hmax")
        self.horizontalSlider_Hmax.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.horizontalSlider_Hmax.setTickInterval(10)
        
        self.horizontalSlider_Smax = QtWidgets.QSlider(self.widget1)
        self.horizontalSlider_Smax.setMaximum(255)
        self.horizontalSlider_Smax.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_Smax.setObjectName("horizontalSlider_Smax")
        self.horizontalSlider_Smax.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.horizontalSlider_Smax.setTickInterval(10)
        
        self.horizontalSlider_Vmax = QtWidgets.QSlider(self.widget1)
        self.horizontalSlider_Vmax.setMaximum(255)
        self.horizontalSlider_Vmax.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_Vmax.setObjectName("horizontalSlider_Vmax")
        self.horizontalSlider_Vmax.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.horizontalSlider_Vmax.setTickInterval(10)

        
        self.horizontalSlider_Hmin.setValue(0)
        self.horizontalSlider_Smin.setValue(0)
        self.horizontalSlider_Vmin.setValue(0)
        self.horizontalSlider_Hmax.setValue(179)
        self.horizontalSlider_Smax.setValue(255)
        self.horizontalSlider_Vmax.setValue(255)
        
        
        self.horizontalSlider_Hmax.valueChanged[int].connect(lambda: self.takeSliderInput(self.horizontalSlider_Hmax))
        self.horizontalSlider_Smax.valueChanged[int].connect(lambda: self.takeSliderInput(self.horizontalSlider_Smax))
        self.horizontalSlider_Vmax.valueChanged[int].connect(lambda: self.takeSliderInput(self.horizontalSlider_Vmax))

        self.verticalLayout_Slidersmax.addWidget(self.horizontalSlider_Hmax)
        self.verticalLayout_Slidersmax.addWidget(self.horizontalSlider_Smax)
        self.verticalLayout_Slidersmax.addWidget(self.horizontalSlider_Vmax)
        
        self.verticalLayout_Sliders.addLayout(self.verticalLayout_Slidersmin)
        self.verticalLayout_Sliders.addLayout(self.verticalLayout_Slidersmax)

        self.widget = QtWidgets.QWidget(self.groupBox_HSVColorSpace)
        self.widget.setGeometry(QtCore.QRect(10, 10, 91, 271))
        self.widget.setObjectName("widget")

        self.verticalLayout_lineEdit = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_lineEdit.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_lineEdit.setObjectName("verticalLayout_lineEdit")

        self.verticalLayout_lineEditmin = QtWidgets.QVBoxLayout()
        self.verticalLayout_lineEditmin.setObjectName("verticalLayout_lineEditmin")

        self.lineEdit_Hmin = QtWidgets.QLineEdit(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_Hmin.setFont(font)
        self.lineEdit_Hmin.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.lineEdit_Hmin.setMaxLength(20)
        self.lineEdit_Hmin.setObjectName("lineEdit_Hmin")
        self.lineEdit_Hmin.placeholderText = "Hmin"

        self.lineEdit_Smin = QtWidgets.QLineEdit(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_Smin.setFont(font)
        self.lineEdit_Smin.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.lineEdit_Smin.setMaxLength(20)
        self.lineEdit_Smin.setObjectName("lineEdit_Smin")
        
        self.lineEdit_Vmin = QtWidgets.QLineEdit(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_Vmin.setFont(font)
        self.lineEdit_Vmin.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.lineEdit_Vmin.setMaxLength(20)
        self.lineEdit_Vmin.setObjectName("lineEdit_Vmin")

        self.lineEdit_Hmin.textEdited.connect(lambda: self.takeTextInput(self.lineEdit_Hmin))
        self.lineEdit_Smin.textEdited.connect(lambda: self.takeTextInput(self.lineEdit_Smin))
        self.lineEdit_Vmin.textEdited.connect(lambda: self.takeTextInput(self.lineEdit_Vmin))

        self.verticalLayout_lineEditmin.addWidget(self.lineEdit_Hmin)
        self.verticalLayout_lineEditmin.addWidget(self.lineEdit_Smin)
        self.verticalLayout_lineEditmin.addWidget(self.lineEdit_Vmin)


        self.verticalLayout_lineEditmax = QtWidgets.QVBoxLayout()
        self.verticalLayout_lineEditmax.setObjectName("verticalLayout_lineEditmax")

        self.lineEdit_Hmax = QtWidgets.QLineEdit(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_Hmax.setFont(font)
        self.lineEdit_Hmax.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.lineEdit_Hmax.setMaxLength(20)
        self.lineEdit_Hmax.setObjectName("lineEdit_Hmax")
        
        self.lineEdit_Smax = QtWidgets.QLineEdit(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_Smax.setFont(font)
        self.lineEdit_Smax.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.lineEdit_Smax.setMaxLength(20)
        self.lineEdit_Smax.setObjectName("lineEdit_Smax")
        
        self.lineEdit_Vmax = QtWidgets.QLineEdit(self.widget)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.lineEdit_Vmax.setFont(font)
        self.lineEdit_Vmax.setLocale(QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates))
        self.lineEdit_Vmax.setMaxLength(20)
        self.lineEdit_Vmax.setObjectName("lineEdit_Vmax")
        
        self.lineEdit_Hmax.textEdited.connect(lambda: self.takeTextInput(self.lineEdit_Hmax))
        self.lineEdit_Smax.textEdited.connect(lambda: self.takeTextInput(self.lineEdit_Smax))
        self.lineEdit_Vmax.textEdited.connect(lambda: self.takeTextInput(self.lineEdit_Vmax))

        self.verticalLayout_lineEditmax.addWidget(self.lineEdit_Hmax)
        self.verticalLayout_lineEditmax.addWidget(self.lineEdit_Smax)
        self.verticalLayout_lineEditmax.addWidget(self.lineEdit_Vmax)

        self.verticalLayout_lineEdit.addLayout(self.verticalLayout_lineEditmin)
        self.verticalLayout_lineEdit.addLayout(self.verticalLayout_lineEditmax)


        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 600, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "HSV Input"))
        self.groupBox_HSVColorSpace.setTitle(_translate("MainWindow", "HSV Color Space"))
        self.pushButton_SaveHSV.setText(_translate("MainWindow", "Save"))

        self.lineEdit_Hmin.setText(_translate("MainWindow", "Hmin:0"))
        self.lineEdit_Smin.setText(_translate("MainWindow", "Smin:0"))
        self.lineEdit_Vmin.setText(_translate("MainWindow", "Vmin:0"))

        self.lineEdit_Hmax.setText(_translate("MainWindow", "Hmax:179"))
        self.lineEdit_Smax.setText(_translate("MainWindow", "Smax:255"))
        self.lineEdit_Vmax.setText(_translate("MainWindow", "Vmax:255"))

    def takeSliderInput(self, slider):
        value = slider.value()
        if slider == self.horizontalSlider_Hmin:
            self.lineEdit_Hmin.setText("Hmin: {}".format(value))
        if slider == self.horizontalSlider_Smin:
            self.lineEdit_Smin.setText("Smin: {}".format(value))
        if slider == self.horizontalSlider_Vmin:
            self.lineEdit_Vmin.setText("Vmin: {}".format(value))
        
        if slider == self.horizontalSlider_Hmax:
            self.lineEdit_Hmax.setText("Hmax: {}".format(value))
        if slider == self.horizontalSlider_Smax:
            self.lineEdit_Smax.setText("Smax: {}".format(value))
        if slider == self.horizontalSlider_Vmax:
            self.lineEdit_Vmax.setText("Vmax: {}".format(value))

    def takeTextInput(self, lineEdit):
        
        text = lineEdit.text()
        try:
            value = int(text[5:])
        except:
            print("incorrect input")
            value = 0
        if lineEdit == self.lineEdit_Hmin:
            if value > 179:
                self.lineEdit_Hmin.setText("Hmin: 179")
            self.horizontalSlider_Hmin.setValue(value)
        if lineEdit == self.lineEdit_Smin:
            if value > 255:
                self.lineEdit_Smin.setText("Smin: 255")
            self.horizontalSlider_Smin.setValue(value)
        if lineEdit == self.lineEdit_Vmin:
            if value > 255:
                self.lineEdit_Vmin.setText("Vmin: 255")
            self.horizontalSlider_Vmin.setValue(value)

        if lineEdit == self.lineEdit_Hmax:
            if value > 179:
                self.lineEdit_Hmax.setText("Hmax: 179")
            self.horizontalSlider_Hmax.setValue(value)
        if lineEdit == self.lineEdit_Smax:
            if value > 255:
                self.lineEdit_Smax.setText("Smax: 255")
            self.horizontalSlider_Smax.setValue(value)
        if lineEdit == self.lineEdit_Vmax:
            if value > 255:
                self.lineEdit_Vmax.setText("Vmax: 255")
            self.horizontalSlider_Vmax.setValue(value)

    def returnHSVInput(self):
        return (self.horizontalSlider_Hmin.value(), self.horizontalSlider_Smin.value(), self.horizontalSlider_Vmin.value(),
               self.horizontalSlider_Hmax.value(), self.horizontalSlider_Smax.value(), self.horizontalSlider_Vmax.value())
        
         
        


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_PopWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
