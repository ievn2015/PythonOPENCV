# -*- coding: utf-8 -*-
"""
Created on Thu May  3 21:56:11 2018

@author: Windows
"""
import cv2
import numpy as np

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
matches = matcher.knnMatch(descs1, descs2, 2)
matches = sorted(matches, key = lambda x:x[0].distance)
good = [m1 for (m1, m2) in matches if m1.distance < 0.23 * m2.distance] #第一幅图0.4才好
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)# 可以都用m.queryIdx
    a = len(dst_pts)
    j = 0
    src_sub_a = []
    dst_sub_a = []
    for i in range(0,a):
        dist = np.linalg.norm(src_pts[i] - dst_pts[i])
        if dist > 140:
            src_sub_a.append(src_pts[i])
            dst_sub_a.append(dst_pts[i])
    cv2.polylines(img1,[np.int32(src_sub_a)],True,(0,255,0),3, cv2.LINE_AA)
    cv2.polylines(img2,[np.int32(dst_sub_a)],True,(0,0,255),3, cv2.LINE_AA)
    matched = np.hstack((img1, img2))
    cv2.imwrite("matched003.png", matched)
    win = cv2.namedWindow('Result', flags=0)
    cv2.imshow('Result', matched)
    cv2.waitKey();
    cv2.destroyAllWindows()
else:
    print( "Not enough matches are found - ".format(len(good),MIN_MATCH_COUNT))