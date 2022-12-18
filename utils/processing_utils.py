import cv2
import numpy as np
import os

class processing_utils(object):

    @staticmethod
    def point_float_to_int(point):
        return (int(point[0]), int(point[1]))

    @staticmethod
    def read_image(file_path):
        return cv2.imread(file_path)

    @staticmethod
    def draw_point(img, point=(0, 0), color = (0, 0, 255)): # red
        point = processing_utils.point_float_to_int(point)
        print(f"get {point=}")
        point_size = 1
        thickness = 4
        return cv2.circle(img, point, point_size, color, thickness)

    @staticmethod
    def draw_line(img, start_point = (0, 0), end_point = (0, 0), color = (0, 255, 0)): # green
        start_point = processing_utils.point_float_to_int(start_point)
        end_point = processing_utils.point_float_to_int(end_point)
        thickness = 3 # width
        return cv2.line(img, start_point, end_point, color, thickness)

    @staticmethod
    def draw_rectangle_by_points(img, left_up=(0, 0), right_down=(0, 0), color = (0, 0, 255)): # red
        left_up = processing_utils.point_float_to_int(left_up)
        right_down = processing_utils.point_float_to_int(right_down)
        thickness = 2 # 寬度 (-1 表示填滿)
        return cv2.rectangle(img, left_up, right_down, color, thickness)

    @staticmethod
    def draw_rectangle_by_xywh(img, xywh=(0, 0, 0, 0), color = (0, 0, 255)): # red
        left_up = processing_utils.point_float_to_int((xywh[0], xywh[1]))
        right_down = processing_utils.point_float_to_int((xywh[0]+xywh[2], xywh[1]+xywh[3]))
        thickness = 2 # 寬度 (-1 表示填滿)
        return cv2.rectangle(img, left_up, right_down, color, thickness)
    
    def select_img(G_list,x,y):
        ROI_list = []
        ROI_sum_list = []     
        list_num = len(G_list)
        for i in range(list_num):
            ROI = G_list[i][y-10:y+10+1,x-10:x+10+1]
            ROI_list.append(ROI)
            ROI_sum = np.sum(ROI)
            ROI_sum_list.append(ROI_sum)
        index = np.argmax(ROI_sum_list)
        
        return index , ROI_list

    def sift_kp(img):
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        sift = cv2.xfeatures2d_SIFT.create()
        kp,des = sift.detectAndCompute(gray_img,None)
        kp_img = cv2.drawKeypoints(gray_img,kp,None) 
        
        return kp_img , kp , des    
    
    def surf_kp(img):
        '''SIFT(surf)特征点检测(速度比sift快)'''
        height, width = img.shape[:2]
        size = (int(width * 0.2), int(height * 0.2))
        shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        gray_img = cv2.cvtColor(shrink,cv2.COLOR_BGR2GRAY)
        surf = cv2.xfeatures2d_SURF.create()
        kp, des = surf.detectAndCompute(gray_img, None)
        
        return kp , des
    
    def get_good_match(img1,img2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)  # k=2,表示寻找两个最近邻

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        out_img1 = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        out_img1[:h1, :w1] = img1
        out_img1[:h2, w1:w1 + w2] = img2
        out_img1 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, out_img1)

        good_match = []
        for m, n in matches:
            if m.distance < 0.85 * n.distance: 
                good_match.append(m)

        out_img2 = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        out_img2[:h1, :w1] = img1
        out_img2[:h2, w1:w1 + w2] = img2
        out_img2 = cv2.drawMatches(img1, kp1, img2, kp2, good_match, out_img2)    
        MIN_MATCH_COUNT = 10

        if len(good_match) > MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_match ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_match ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()

            pts = np.float32([ [0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            dst = np.int32(dst)
            img2 = cv2.polylines(img2,[dst],True,0,0, cv2.LINE_AA)

        else:
            print( "Not enough matches are found - {}/{}".format(len(good_match), MIN_MATCH_COUNT) )
            matchesMask = None
            
        imgOut = img2
        
        return img1 , img2 , imgOut , dst
        
    def position_transform(img , index , position):
        height , width = img.shape[:2]
        delta_x1 , delta_y1 , delta_x2 , delta_y2 = np.abs(position)
        new_img = img[delta_y1:height-delta_y2,delta_x1:width-delta_x2]

        new_img = cv2.resize(new_img,(width,height))
        
        return new_img

    def resize (img):
        height , width , channel = img.shape
        if height > width:
            img_resize = cv2.resize(img,(600,800))
        elif width > height:
            img_resize = cv2.resize(img,(800,600))
        else:
            img_resize = cv2.resize(img,(600,600))
        return img_resize

    def gradient (img):
        img_gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img_gray,(3,3),1)
        Laplacian = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]])
        Laplacian_result = cv2.filter2D(blur, ddepth = -1, kernel = Laplacian)
        
        return Laplacian_result
        
    def sharpen (img):
        blur = cv2.GaussianBlur(img,(3,3),1)
        Sharpen = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
        Sharpen_result = cv2.filter2D(blur, ddepth = -1, kernel = Sharpen)
        
        return Sharpen_result
    
    def img_path_processing(img_path , x , y):
        total = []
        img_list = []
        Gradient_list = []
        Sharpen_list = []
        new_img_list = []
        dst_list = []
        
        all_content = os.listdir(img_path)
        for file in all_content :
            if file.endswith(".JPG") or file.endswith(".jpg") or file.endswith(".PNG") or file.endswith(".png"):
                total.append(file)
        t = len(total)
        
        img_background = cv2.imread(img_path + total[0])
        img_background = processing_utils.resize(img_background)
        Sharpen_background = processing_utils.sharpen(img_background)
        
        
        for num_img in range(t):
            img = cv2.imread(img_path + total[num_img])

            img_resize = processing_utils.resize(img)     
            img_list.append(img_resize)
            height , width , channel = img_resize.shape
                    
            gradient_result = processing_utils.gradient(img_resize)
            Sharpen_result = processing_utils.sharpen(img_resize)
            Gradient_list.append(gradient_result)
            Sharpen_list.append(Sharpen_result)
            
            img1 , img2 , imgOut , dst1 = processing_utils.get_good_match(Sharpen_list[num_img] , Sharpen_background)
            new_img_list.append(imgOut)
            dst_list.append(dst1)

        index , ROI_list = processing_utils.select_img(Gradient_list,int(x),int(y))
        img_change = img_list[index]

        height , width = img_change.shape[:2]
        img1 , img2 , imgOut , dst1 = processing_utils.get_good_match(Sharpen_list[index],Sharpen_list[0])
        delta_x1 = np.floor((dst_list[index][0][0][0] + dst_list[index][0][0][1]) / 2)
        delta_y1 = np.floor((dst_list[index][0][0][1] + dst_list[index][3][0][1]) / 2)
        delta_x2 = dst_list[index][2][0][0] - width
        delta_y2 = dst_list[index][2][0][1] - height
        position = np.abs([delta_x1 , delta_y1 , delta_x2 , delta_y2]).astype(np.uint8)
        new_img = processing_utils.position_transform(img_change,index,position)

            
        return new_img , index + 1
    