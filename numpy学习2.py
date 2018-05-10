# -*- coding: utf-8 -*-
"""
Created on Wed May  9 21:18:17 2018

@author: Windows
"""
import cv2
import numpy as np
imgname1 = "003-a.jpg"
imgname2 = "003-b.jpg"

img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)
sss=np.zeros([480,640],dtype=np.uint8) 
print (sss)
sss[300:350,310:400]=255
print (sss)
image=cv2.add(img1, np.zeros(np.shape(img1), dtype=np.uint8), mask=sss)
cv2.imwrite("mask.png", image)
cv2.imshow('Result', image)

# 转换格式d = tuple(map(tuple, src_pts))