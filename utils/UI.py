from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1280, 800)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(60, 50, 802, 602))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.scrollArea = QtWidgets.QScrollArea(self.verticalLayoutWidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        # self.scrollAreaWidgetContents = QtWidgets.QWidget()
        # self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 937, 527))
        # self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.label_img = QtWidgets.QLabel()
        self.label_img.setGeometry(QtCore.QRect(0, 0, 802, 602))
        self.label_img.setObjectName("label_img")
        self.scrollArea.setWidget(self.label_img)

        self.verticalLayout.addWidget(self.scrollArea)
        # self.btn_zoom_in = QtWidgets.QPushButton(self.centralwidget)
        # self.btn_zoom_in.setGeometry(QtCore.QRect(190, 680, 89, 25))
        # self.btn_zoom_in.setObjectName("btn_zoom_in")
        self.label_img_shape = QtWidgets.QLabel(self.centralwidget)
        self.label_img_shape.setGeometry(QtCore.QRect(900, 670, 641, 21))
        self.label_img_shape.setObjectName("label_img_shape")
        self.btn_open_file = QtWidgets.QPushButton(self.centralwidget)
        self.btn_open_file.setGeometry(QtCore.QRect(60, 680, 89, 25))
        self.btn_open_file.setObjectName("btn_open_file")
        self.label_file_name = QtWidgets.QLabel(self.centralwidget)
        self.label_file_name.setGeometry(QtCore.QRect(40, 910, 941, 31))
        self.label_file_name.setObjectName("label_file_name")
        # self.slider_zoom = QtWidgets.QSlider(self.centralwidget)
        # self.slider_zoom.setGeometry(QtCore.QRect(290, 680, 231, 21))
        # self.slider_zoom.setProperty("value", 50)
        # self.slider_zoom.setOrientation(QtCore.Qt.Horizontal)
        # self.slider_zoom.setObjectName("slider_zoom")
        self.label_ratio = QtWidgets.QLabel(self.centralwidget)
        self.label_ratio.setGeometry(QtCore.QRect(650, 680, 641, 21))
        self.label_ratio.setObjectName("label_ratio")
        # self.btn_zoom_out = QtWidgets.QPushButton(self.centralwidget)
        # self.btn_zoom_out.setGeometry(QtCore.QRect(540, 680, 89, 25))
        # self.btn_zoom_out.setObjectName("btn_zoom_out")
        self.label_click_pos = QtWidgets.QLabel(self.centralwidget)
        self.label_click_pos.setGeometry(QtCore.QRect(900, 600, 191, 20))
        self.label_click_pos.setObjectName("label_click_pos")
        self.label_real_pos = QtWidgets.QLabel(self.centralwidget)
        self.label_real_pos.setGeometry(QtCore.QRect(900, 700, 191, 20))
        self.label_real_pos.setObjectName("label_real_pos")
        self.label_norm_pos = QtWidgets.QLabel(self.centralwidget)
        self.label_norm_pos.setGeometry(QtCore.QRect(900, 680, 191, 20))
        self.label_norm_pos.setObjectName("label_norm_pos")
        # self.text_ratio_roi = QtWidgets.QTextEdit(self.centralwidget)
        # self.text_ratio_roi.setGeometry(QtCore.QRect(130, 700, 191, 71))
        # self.text_ratio_roi.setObjectName("text_ratio_roi")
        # self.text_real_roi = QtWidgets.QTextEdit(self.centralwidget)
        # self.text_real_roi.setGeometry(QtCore.QRect(410, 700, 201, 71))
        # self.text_real_roi.setObjectName("text_real_roi")
        # self.label_info_ratio_roi = QtWidgets.QLabel(self.centralwidget)
        # self.label_info_ratio_roi.setGeometry(QtCore.QRect(60, 680, 91, 31))
        # self.label_info_ratio_roi.setObjectName("label_info_ratio_roi")
        # self.label_info_real_roi = QtWidgets.QLabel(self.centralwidget)
        # self.label_info_real_roi.setGeometry(QtCore.QRect(350, 700, 91, 31))
        # self.label_info_real_roi.setObjectName("label_info_real_roi")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1280, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_img.setText(_translate("MainWindow", "label_img"))
        # self.btn_zoom_in.setText(_translate("MainWindow", "zoom in"))
        self.label_img_shape.setText(_translate("MainWindow", "Current image shape: (0,0), Origin image shape: (0,0)"))
        self.btn_open_file.setText(_translate("MainWindow", "Open file"))
        self.label_file_name.setText(_translate("MainWindow", "file name:"))
        self.label_ratio.setText(_translate("MainWindow", "ratio: 100%"))
        # self.btn_zoom_out.setText(_translate("MainWindow", "zoom out"))
        self.label_click_pos.setText(_translate("MainWindow", "clicked position = (x, y)"))
        self.label_real_pos.setText(_translate("MainWindow", "real position = (x, y)"))
        self.label_norm_pos.setText(_translate("MainWindow", "normalized position = (x, y)"))
        # self.label_info_ratio_roi.setText(_translate("MainWindow", "Ratio ROI:"))
        # self.label_info_real_roi.setText(_translate("MainWindow", "Real ROI:"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
