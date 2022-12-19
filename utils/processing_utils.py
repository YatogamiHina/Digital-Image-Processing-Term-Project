import cv2
import numpy as np

class processing_utils(object):
    def read_image(file_path):
        return cv2.imread(file_path)
    
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
    
    def get_good_match(img1,img2):
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(des1, des2, k=2)  # KNN

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        out_img1 = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        out_img1[:h1, :w1] = img1
        out_img1[:h2, w1:w1 + w2] = img2

        good_match = []
        for m, n in matches:
            if m.distance < 0.85 * n.distance: 
                good_match.append(m)

        out_img2 = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
        out_img2[:h1, :w1] = img1
        out_img2[:h2, w1:w1 + w2] = img2 
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
            
        # imgOut = img2
        
        return dst
        

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
    
    
    def img_sort(data_list , dst_temp_list):
        img_list = []
        t = len(data_list)
        sort_list = sorted(range(len(dst_temp_list)), key=lambda k: dst_temp_list[k], reverse=True)
        for i in range(t):
            img_list.append(data_list[sort_list[i]])
            
        img_background = img_list[-1]
        Sharpen_background = processing_utils.sharpen(img_background)
        dst_list = []

        for i in range(t):
            Sharpen_result = processing_utils.sharpen(img_list[i])
            dst1 = processing_utils.get_good_match(Sharpen_result , Sharpen_background)
            dst_list.append(dst1)
        
        return img_list , dst_list , img_background , Sharpen_background
    
    def position_transform(img , position):
        height , width = img.shape[:2]
        delta_x1 , delta_y1 , delta_x2 , delta_y2 = np.abs(position)
        new_img = img[delta_y1:height-delta_y2,delta_x1:width-delta_x2]

        new_img = cv2.resize(new_img,(width,height))
        
        return new_img
    
    def img_registration(img_list,dst_list):
        new_img_list = []
        gradient_list = []
        t = len(img_list)
        height , width = img_list[0].shape[:2] 
        # img_background = img_list[-1]
        for i in range(t):
            delta_x1 = np.floor((dst_list[i][0][0][0] + dst_list[i][0][0][1]) / 2)
            delta_y1 = np.floor((dst_list[i][0][0][1] + dst_list[i][3][0][1]) / 2)
            delta_x2 = dst_list[i][2][0][0] - width
            delta_y2 = dst_list[i][2][0][1] - height
            position = np.abs([delta_x1 , delta_y1 , delta_x2 , delta_y2]).astype(np.uint8)
            new_img = processing_utils.position_transform(img_list[i],position)
            new_img_list.append(new_img)
            gradient_result = processing_utils.gradient(img_list[i])
            gradient_list.append(gradient_result)

            
        return new_img_list , gradient_list
