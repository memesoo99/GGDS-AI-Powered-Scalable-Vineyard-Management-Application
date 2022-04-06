import cv2
from detectron2.utils.visualizer import GenericMask
import pycocotools.mask as mask_util
import pickle
import os
import pandas as pd
import numpy as np

# 모든 변수들을 구하는!!!
# contour에서 이미지 외곽ㅗ 잡는ㅕㅇ우 있으니 다 넘겨주면 ㅇ나됨
# run on mask -> pkl로 저장, csv로 저장 둘다 넘겨주기

class Contours():
    def __init__(self,pkl_path, image_path,df_path=None):
        
        if not os.path.exists(df_path):
            self.df = pd.DataFrame(columns = ["image","number of instances","sunburn_ratio","diameter","circularity","density","aspect ratio","grade"])
        else:
            self.df = pd.read_csv(df_path)
        
        # print(self.df)
        
        with open(pkl_path,"rb") as f:
        # with open("/workspace/regression_data/masks/20170227_130321_HDR_masks.pkl","rb") as f:
            self.mask_n_class = pickle.load(f)
        self.df_path = df_path
        self.masks = self.mask_n_class[0]
        # mask_util.decode(rle)[:, :]
        self.image_path = image_path
        self.image_shape = self.masks[0].shape
        self.image_size = self.image_shape[0] * self.image_shape[1]
        self.feat_area_sum = 0.0
        self.feat_diameter_sum = 0.0
        self.feat_num_instances = len(self.masks)
        self.sick_num_instances = self.mask_n_class[1]
        self.healthy_num_instances = self.feat_num_instances - self.sick_num_instances
        self.hue_list=[]
        self.feat_diameter = 0.0
        self.feat_circularity = 0
        self.contour_size_list = []

        self.test = np.zeros_like(self.masks[0])
        
    def run(self):
        for i in range(len(self.masks)):
            # rle_decode = mask_util.decode(self.masks[i])[:, :]
            # print(np.array_equal(self.masks[i], self.masks[i+1], equal_nan=False)) mask 잘 들어가는데??
            contours = self.mask_to_contour(self.masks[i]) # H,W,3 ndarray
        
            # print(self.masks[i].shape)
            # contour = self.mask_to_contour(rle_decode) # H,W,3 ndarray
            # for contour in contours:
            contour_area = self.area(contours)
            # print("contour_area:",contour_area)
            contour_perimeter = self.perimeter(contours)

            # contour_aspect_ratio = self.aspect_ratio(contour)
            if contour_perimeter == 0:
                self.feat_num_instances-=1
                continue
            hue_extraction = self.hue_extraction(contours)
            self.hue_list.append(hue_extraction)
            contour_circularity = self.circularity(contour_area,contour_perimeter)
            contour_diameter = self.circles(contours)
                
            self.feat_circularity+=contour_circularity
            self.feat_area_sum +=contour_area
            self.feat_diameter_sum +=contour_diameter
            self.contour_size_list.append(contour_area)
            
        self.feature_extraction()
        
    
    def feature_extraction(self):
        self.feat_circularity = self.feat_circularity / self.feat_num_instances
        self.feat_diameter = self.feat_diameter_sum/ self.feat_area_sum
        self.density = self.feat_area_sum / self.image_size
        # print(self.image_shape[0], self.image_shape[1])

        self.aspect_ratio = float(self.image_shape[0]) / float(self.image_shape[1])
        # print("aspect_ratio",self.aspect_ratio)
        self.grade = self.get_grade()
        # print(self.grade)
        self.ave_hue = np.mean(self.hue_list)
        self.df = self.df.append({'image' : self.image_path , 'number of instances' : self.feat_num_instances, 'sunburn_ratio': self.sunburn_ratio , 'diameter' : self.feat_diameter, 'circularity':self.feat_circularity, 'density':self.density, 'aspect ratio': self.aspect_ratio,'average_hue':self.ave_hue, 'grade': self.grade} , ignore_index=True)
        self.df.to_csv(self.df_path, index = False)
        # print({'image' : self.image_path , 'number of instances' : self.feat_num_instances, 'sunburn_ratio': self.sunburn_ratio , 'diameter' : self.feat_diameter, 'circularity':self.feat_circularity, 'density':self.density, "aspect ratio": self.aspect_ratio, "grade": self.grade})
        
    def select_second(contours):
        areas = []
        for i in range(len(contours)):
            contour_area = self.area(contours[i])
            areas.append(contour_area)
        sort_index = numpy.argsort(areas)
        return contours[sort_index[-2]]

    def mask_to_contour(self, mask):
       
        backtorgb = mask
        
        backtorgb[ mask > 0] = 255 
        backtorgb.astype(np.uint8) 
        ret, img_binary = cv2.threshold(backtorgb, 127, 255, cv2.THRESH_BINARY_INV)
    
        contours, _ = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        areas = []
        for i in range(len(contours)):
                if len(contours)==1:
                    contour_area = self.area(contours[i])
                    areas.append(contour_area)
                    index = -1
                    break
                contour_area = self.area(contours[i])
                areas.append(contour_area)
                index = -2
        sort_index = np.argsort(areas)

        return contours[sort_index[index]]
        

    def hue_extraction(self,contours):
        img = cv2.imread(self.image_path)
        # print(self.image_path)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue = hsv[:,:,0]
        lower_range = np.array([75, 50, 50])
        upper_range = np.array([165, 200, 255])
        thresh = cv2.inRange(hsv, lower_range, upper_range)
        N2=0
        count = img.copy()
        cv2.drawContours(count, contours, 0, (100, 0, 255), 2)
        mask = np.zeros_like(thresh, dtype=np.uint8)
        cv2.drawContours(mask, contours, 0, (255,255,255), -1)
        ave_hue = np.mean(hue[np.where(mask==255)])
        return ave_hue



    def get_grade(self): 
        # 낱알의 고르기
        # https://scienceon.kisti.re.kr/srch/selectPORSrchReport.do?cn=TRKO201100016401&dbt=TRKO
        # Berry Hue 편차
        # color variety 고려
        # https://www.ri.cmu.edu/pub_files/2016/8/pothen_ifac_2016.pdf
        # sunburn, we call it sick berry
        # https://unece.org/fileadmin/DAM/trade/agr/standard/standard/fresh/FFV-Std/English/19_TablesGrapes.pdf
        # 위 문서 내용 살짝 변형시켜서 3등급으로 나눔. 
        # Extra class - no defects
        # class I - slight
        # class II - average
        # waste - class II 이하

        self.sunburn_ratio = self.sick_num_instances/self.feat_num_instances
        if self.sunburn_ratio == 0.0:
            self.sunburn_grade = 0
        elif self.sunburn_ratio > 0 and self.sunburn_ratio < 0.05:
            self.sunburn_grade = 1
        elif self.sunburn_ratio >= 0.05 and self.sunburn_ratio < 0.2:
            self.sunburn_grade = 2
        else:
            self.sunburn_grade = 3

        uniformity = np.std(self.contour_size_list) # 사이즈 통일성 .. 가려진거 때문에 살짝 애매하긴한데 그렇게 널널하게 기준 잡아서
        if uniformity <1.5:
            uniformity_grade = 0
        else:
            uniformity_grade = 1
        hue_uniformity = np.std(self.hue_list)
        if hue_uniformity < 25.0:
            hue_grade = 0
        else:
            hue_grade = 1
        # print("HUE:",hue_uniformity)
        # if hue_uniformity 
        grade = (self.sunburn_grade*2 + uniformity_grade*1 + hue_grade) // 3 
        return grade


    def area(self, contour):
        return cv2.contourArea(contour)
    
    def perimeter(self,contour):
        return cv2.arcLength(contour,True)

    def aspect_ratio(self,contour): #종횡비
        x,t,w,h = cv2.boundingRect(contour)
        
        return float(x)/h

    def circularity(self,area,parameter):
        cir = (4*3.14*area)/(parameter*parameter)
        if cir < 0.7:
            return 0 #원이 아님
        else:
            return 1 #원임

    def circles(self,cnt): #외접하는 가장 작은 원
        (x,y),r = cv2.minEnclosingCircle(cnt)
        center = (int(x),int(y))
        r = int(r)
        return r
    
    def oval(self,cnt):
        elipse = cv2.fitEllopse(cnt)
        # [vx,vy,x,y] = cv2




# features = Contours(f"/workspace/regression_data/masks/20170227_130321_HDR_masks.pkl", f"/workspace/regression_data/images3/20170227_130321_HDR.jpg",'/workspace/features_temp.csv')
# features.run()