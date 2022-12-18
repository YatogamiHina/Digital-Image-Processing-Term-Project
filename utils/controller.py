from PyQt5 import QtCore 
# from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QFileDialog
# from PyQt5.QtCore import QThread, pyqtSignal


from utils.UI import Ui_MainWindow
from utils.img_controller import img_controller

class MainWindow_controller(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.set_init_ui()
        
    def set_init_ui(self):
        self.ui.btn_open_file.clicked.connect(self.open_file)     
   
    
    def open_file(self):    
        folder_path = QFileDialog.getExistingDirectory(self, "Open folder", "./") + '/' 
        self.img_controller = img_controller(folder_path, self.ui)        
  