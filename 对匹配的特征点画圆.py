# -*- coding: utf-8 -*-
"""
Created on Mon May  7 19:41:10 2018
图像的分块搜索
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

matches = matcher.knnMatch(descs1, descs2, 2)
matches = sorted(matches, key = lambda x:x[0].distance)
good = [m1 for (m1, m2) in matches if m1.distance < 0.9 * m2.distance] #第一幅图0.4才好
if len(good) > MIN_MATCH_COUNT:
    src_pts = np.int32([kpts1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.int32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)# 可以都用m.queryIdx
    a = len(src_pts)/2
    a = int (a)
    print(a,len(src_pts))
    c = src_pts.ravel()
    #c = c.reshape(2,1)
    #c = np.tuple()
    canvas = img1.copy()
    print (c[0],c[1])
    for k in range(1,a):
        
        cv2.circle(canvas, (c[2*k],c[2*k-1]), 80, (255, 255, 255), -1)
    
    #cv2.polylines(img1,[src_pts[0]],True,(0,255,0),3, cv2.LINE_AA)
    #for row in src_pts: 
        #a = src_pts
        #print (a)
        #cv2.circle(img1,a , 63, (255, 0, 0), -1)

    matched = np.hstack((canvas, img2))
    cv2.imwrite("mask.png", matched)
    win = cv2.namedWindow('Result', flags=0)
    cv2.imshow('Result', matched)
    cv2.waitKey();
    cv2.destroyAllWindows()
else:
    print( "Not enough matches are found - ".format(len(good),MIN_MATCH_COUNT))