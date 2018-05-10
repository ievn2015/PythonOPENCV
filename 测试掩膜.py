# -*- coding: utf-8 -*-
"""
Created on Wed May  9 23:17:36 2018

@author: Windows
"""
import cv2
import numpy as np

img1 = cv2.imread('003-b.jpg')
img2 = cv2.imread('canvas.png')

# 把logo放在左上角，所以我们只关心这一块区域
rows, cols = img2.shape[:2]
roi = img1[:rows, :cols]

# 创建掩膜
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# 保留除logo外的背景
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
dst = cv2.add(img1_bg, img2)  # 进行融合
img1[:rows, :cols] = dst  # 融合后放在原图上
cv2.imwrite("test.png", img1)
win = cv2.namedWindow('Result', flags=0)
cv2.imshow('Result', img1)
cv2.waitKey();
cv2.destroyAllWindows()
