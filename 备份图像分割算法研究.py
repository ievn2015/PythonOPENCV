# -*- coding: utf-8 -*-
"""
Created on Wed May 23 23:57:27 2018

@author: Windows
"""

import cv2  
import numpy as np  
import matplotlib.pyplot as plt  
  
  
# 使用2g-r-b分离土壤与背景  
  
src = cv2.imread('003-b.jpg')
cv2.imshow('src', src)  
  
# 转换为浮点数进行计算  
fsrc = np.array(src, dtype=np.float32) / 255.0  
(b,g,r) = cv2.split(fsrc)  
#gray=cv2.cvtColor(src,cv2.COLOR_BGR2GRAY)
gray = 2*r  - b - g  
  
# 求取最大值和最小值  
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)  
  
# 计算直方图  
hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])  
plt.plot(hist)  
plt.show()  
gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)  
(thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)  
cv2.imshow('bin_img', bin_img)  
(b8, g8, r8) = cv2.split(src)  
color_img = cv2.merge([b8 & bin_img, g8 & bin_img, r8 & bin_img]) 
cv2.imshow('color_img', color_img)  
# 获取轮廓，我们的目的是分块，因此只使用外层轮廓，使用点序列的形式  



cv2.waitKey()  