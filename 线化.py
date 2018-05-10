# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 20:33:11 2018

@author: Windows
"""

#!/usr/bin/python3
# 2017.11.02 17:31:24 CST
# 2017.11.02 17:51:13 CST
import cv2
import numpy as np
img = cv2.imread("003-b.jpg")

## BGR => Gray； 高斯滤波； Canny 边缘检测
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gaussed = cv2.GaussianBlur(gray, (3,3), 0)
cannyed = cv2.Canny(gaussed, 10, 220)

## 将灰度边缘转化为BGR 
cannyed2 = cv2.cvtColor(cannyed, cv2.COLOR_GRAY2BGR) 

## 创建彩色边缘 
mask = cannyed > 0             # 边缘掩模
canvas = np.zeros_like(img)    # 创建画布
canvas[mask] = img[mask]       # 赋值边缘

## 保存
res = np.hstack((img, cannyed2, canvas))   #　组合在一起　
cv2.imwrite("found.png", res)


## 显示 
cv2.imshow("canny in opencv ", res)        

# 保持10s, 等待按键响应（超时或按键则进行下一步）
key = 0xFF & cv2.waitKey(1000*10)
if key in (ord('Q'), ord('q'), 27):
    ## 这部分用作演示用
    print("Exiting...")

## 销毁窗口
cv2.destroyAllWindows()