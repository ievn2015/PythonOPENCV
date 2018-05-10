# -*- coding: utf-8 -*-
"""
Created on Thu May  3 21:56:11 2018

@author: Windows
"""

import cv2
import numpy as np

"""
使用Sift特征点检测和匹配查找场景中特定物体。
"""

MIN_MATCH_COUNT = 4 

imgname1 = "003-a.jpg"
imgname2 = "003-b.jpg"

img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kpts1, descs1 = sift.detectAndCompute(gray1, None)
kpts2, descs2 = sift.detectAndCompute(gray2, None)
b = len(kpts1)
c = len(kpts2)
k = 0
n = 0
kp1 = []
kp2 = []
for k in range(0,b):
    kp1.append(kpts1[k].pt)
for n in range(0,c):
    kp2.append(kpts2[n].pt)
kp1 = np.float32(kp1)
kp2 = np.float32(kp2)
print(kp1)
matched = np.hstack((img1, img2))
cv2.imwrite("matched.png", matched)
win = cv2.namedWindow('test win2', flags=0)
cv2.imshow('test win2', matched)
cv2.waitKey();
cv2.destroyAllWindows()

