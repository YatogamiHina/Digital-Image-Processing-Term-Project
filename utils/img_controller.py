from PyQt5 import QtCore 
from PyQt5.QtGui import QImage, QPixmap
import os
import cv2
import numpy as np
from utils.processing_utils import processing_utils
import base64
from utils.sad_panda import explode
from io import BytesIO
from PIL import Image, ImageQt

class img_controller(object):
    def __init__(self, target_path, ui):
        total = []
        all_content = os.listdir(target_path)
        for file in all_content :
            if file.endswith(".JPG") or file.endswith(".jpg") or file.endswith(".PNG") or file.endswith(".png"):
                total.append(file)
        t = len(total)
        self.total = total
        
        if t > 0:
            img_path = target_path + total[0]
            self.img_background = processing_utils.read_image(img_path)
            self.img_background = processing_utils.resize(self.img_background)
            self.Sharpen_background = processing_utils.sharpen(self.img_background)
            self.target_path = target_path
            self.img_path = img_path
            
        else:
            byte_data = base64.b64decode(explode)
            image_data = BytesIO(byte_data)
            self.origin_img = cv2.imdecode(np.frombuffer(image_data.getvalue(), np.uint8), 1)
            self.img_path = target_path
            
        self.ui = ui
        self.ratio_value = 50
        self.read_file_and_init()

    def read_file_and_init(self):
        img_list = []
        Gradient_list = []
        Sharpen_list = []
        new_img_list = []
        dst_list = []
        t = len(self.total)

        

        # try:     
        if t > 0:    
            for num_img in range(t):
                img_path = self.target_path + self.total[num_img]
                img1 = processing_utils.read_image(img_path)
                # print(img_path)
                # height , width , channel = img.shape
                img_resize = processing_utils.resize(img1)     
                img_list.append(img_resize)
                height , width , channel = img_resize.shape
                # cv2.namedWindow("background")
                # cv2.imshow('original' , img_resize)           
                    
                gradient_result = processing_utils.gradient(img_resize)
                Sharpen_result = processing_utils.sharpen(img_resize)
                # cv2.imwrite(target_path + 'sharpen/Sharpen_result_' + str(num_img+1) + '.jpg' , Sharpen_result)
                Gradient_list.append(gradient_result)
                Sharpen_list.append(Sharpen_result)
                
                img1 , img2 , imgOut , dst1 = processing_utils.get_good_match(Sharpen_list[num_img] , self.Sharpen_background)
                new_img_list.append(imgOut)
                dst_list.append(dst1)

            self.img_list = img_list
            self.Gradient_list = Gradient_list
            self.Sharpen_list = Sharpen_list
            self.new_img_list = new_img_list
            self.dst_list = dst_list

            self.origin_img = self.img_background
            self.origin_height, self.origin_width, self.origin_channel = self.origin_img.shape
        # except:
        else:
            # self.origin_img = processing_utils.read_image('sad_panda.jpg')
            self.origin_height, self.origin_width, self.origin_channel = self.origin_img.shape

        self.display_img = self.origin_img
        self.__update_text_file_path()
        self.ratio_value = 50 # re-init
        self.__update_img()
        self.t = t

    def set_path(self, img_path):
        self.img_path = img_path
        self.read_file_and_init()


    def __update_img_ratio(self):
        self.ratio_rate = pow(10, (self.ratio_value - 50)/50)
        qpixmap_height = self.origin_height * self.ratio_rate
        self.qpixmap = self.qpixmap.scaledToHeight(qpixmap_height)
        self.__update_text_ratio()
        self.__update_text_img_shape()

    def __update_img(self):       
        bytesPerline = 3 * self.origin_width
        qimg = QImage(self.display_img, self.origin_width, self.origin_height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.qpixmap = QPixmap.fromImage(qimg)
        self.__update_img_ratio()
        self.ui.label_img.setPixmap(self.qpixmap)
        self.ui.label_img.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.ui.label_img.mousePressEvent = self.set_clicked_position

    def __update_text_file_path(self):
        self.ui.label_file_name.setText(f"File path = {self.img_path}")

    def __update_text_ratio(self):
        self.ui.label_ratio.setText(f"{int(100*self.ratio_rate)} %")

    def __update_text_img_shape(self):
        current_text = f"Current img shape = ({self.qpixmap.width()}, {self.qpixmap.height()})"
        origin_text = f"Origin img shape = ({self.origin_width}, {self.origin_height})"
        self.ui.label_img_shape.setText(current_text+"\t"+origin_text)

    def __update_text_clicked_position(self, x, y):
        # give me qpixmap point
        self.ui.label_click_pos.setText(f"Clicked postion = ({x}, {y})")
        norm_x = x/self.qpixmap.width()
        norm_y = y/self.qpixmap.height()
        # print(f"(x, y) = ({x}, {y}), normalized (x, y) = ({norm_x}, {norm_y})")
        self.ui.label_norm_pos.setText(f"Normalized postion = ({norm_x:.3f}, {norm_y:.3f})")
        self.ui.label_real_pos.setText(f"Real postion = ({int(norm_x*self.origin_width)}, {int(norm_y*self.origin_height)})")

    def set_zoom_in(self):
        self.ratio_value = max(0, self.ratio_value - 1)
        self.__update_img()

    def set_zoom_out(self):
        self.ratio_value = min(100, self.ratio_value + 1)
        self.__update_img()

    def set_slider_value(self, value):
        self.ratio_value = value
        self.__update_img()

    def set_clicked_position(self, event):
        x = event.pos().x()
        y = event.pos().y()
        self.__update_text_clicked_position(x, y)
        if self.t > 0:
            self.img_path_processing(x, y)  
        
    def img_path_processing(self , x , y):
        index , ROI_list = processing_utils.select_img(self.Gradient_list,int(x),int(y))
        img_change = self.img_list[index]
        height , width = img_change.shape[:2]
        img1 , img2 , imgOut , dst1 = processing_utils.get_good_match(self.Sharpen_list[index],self.Sharpen_list[0])
        delta_x1 = np.floor((self.dst_list[index][0][0][0] + self.dst_list[index][0][0][1]) / 2)
        delta_y1 = np.floor((self.dst_list[index][0][0][1] + self.dst_list[index][3][0][1]) / 2)
        delta_x2 = self.dst_list[index][2][0][0] - width
        delta_y2 = self.dst_list[index][2][0][1] - height
        position = np.abs([delta_x1 , delta_y1 , delta_x2 , delta_y2]).astype(np.uint8)
        new_img = processing_utils.position_transform(img_change,index,position)
        self.display_img = new_img
        self.index = index
        self.__update_img()
 