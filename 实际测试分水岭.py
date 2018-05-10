# -*- coding: utf-8 -*-
"""
Created on Wed May  9 23:17:36 2018

@author: Windows
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
MIN_MATCH_COUNT = 4
imgname1 = "003-a.jpg"
imgname2 = "003-b.jpg"
img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
matcher = cv2.FlannBasedMatcher(dict(algorithm= 1, trees = 5), {})
kpts1, descs1 = sift.detectAndCompute(gray1, None,)
kpts2, descs2 = sift.detectAndCompute(gray2, None,)

matches = matcher.knnMatch(descs1, descs2, 2)
matches = sorted(matches, key = lambda x:x[0].distance)
good = [m1 for (m1, m2) in matches if m1.distance < 0.9 * m2.distance] #第一幅图0.4才好
print(len(good))
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.int32([kpts1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.int32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)# 可以都用m.queryIdx
    a = len(src_pts)/2
    a = int (a)
    print(a,len(src_pts))
    c = src_pts.ravel()
    d = dst_pts.ravel()
    #c = c.reshape(2,1)
    #c = np.tuple()
    canvas1 = gray1.copy()
    canvas2 = gray2.copy()
    #print (c[0],c[1])
    for k in range(1,a):
        cv2.circle(canvas1, (c[2*k],c[2*k-1]), 80, (0, 0, 0), -1)
        cv2.circle(canvas2, (d[2*k],d[2*k-1]), 80, (255, 255, 255), -1)

ret, thresh = cv2.threshold(gray1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
# 降噪处理
kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
# 确定背景
sure_bg = cv2.dilate(opening,kernel,iterations=3)
# 查找前景
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# 查找未确定区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)
# 标注
ret, markers = cv2.connectedComponents(sure_fg)
markers = markers+1
# 将未确定区域置为0
markers[unknown==255] = 0
# 执行分水岭
markers = cv2.watershed(img1,markers)
img1[markers == -1] = [255,0,0]
cv2.imshow("img",img1)
cv2.waitKey()
cv2.destroyAllWindows()