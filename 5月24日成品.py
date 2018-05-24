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

MIN_MATCH_COUNT = 4  # 至少需要4个健壮特征点

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

good = [m1 for (m1, m2) in matches if m1.distance < 0.5 * m2.distance]
#print(len(good))
canvas = img2.copy()

if len(good) > MIN_MATCH_COUNT:

    src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) #获取匹配到的坐标存在np中第一张图
    dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2) #获取匹配到的坐标存在np中第二张图
    #print (src_pts[0],dst_pts[0])
    #a = src_pts[0]
    #b = dst_pts[0]
    #c = np.sqrt(np.sum(np.square(a - b) 
    a = len(dst_pts)
    j = 0
    src_sub_a = []
    dst_sub_a = []
    for i in range(0,a):
        dist = np.linalg.norm(src_pts[i] - dst_pts[i])
        if dist > 150:
            src_sub_a.append(src_pts[i])
            dst_sub_a.append(dst_pts[i])
    src_sub_a = np.int32(src_sub_a)
    dst_sub_a = np.int32(dst_sub_a)
    
    j = len(src_sub_a)
    src_sub_a = src_sub_a[0:j,0]
    print (src_sub_a)
    X = src_sub_a[:,0]
    maxX = max(X)
    minX = min(X)
    print (maxX)
    print (X)
    Y = src_sub_a[:,1]
    maxY = max(Y)
    minY = min(Y)
    cv2.rectangle(img1, (minX, minY), (maxX, maxY), (0, 255, 0), 20)
    
    j = len(dst_sub_a)
    dst_sub_a = dst_sub_a[0:j,0]
    print (dst_sub_a)
    X1 = dst_sub_a[:,0]
    maxX1 = max(X1)
    minX1 = min(X1)
    print (maxX1)
    print (X1)
    Y1 = src_sub_a[:,1]
    maxY1 = max(Y1)
    minY1 = min(Y1)
    cv2.rectangle(canvas, (minX1, minY1), (maxX1, maxY1), (0, 0, 255), 20)
    
    #print(X)
    #print (src_sub_a ,dst_sub_a)
    #cv2.polylines(img1,[np.int32(src_sub_a)],True,(0,255,0),3, cv2.LINE_AA)
    #cv2.polylines(canvas,[np.int32(dst_sub_a)],True,(0,0,255),3, cv2.LINE_AA)
    matched = np.hstack((img1, canvas))   #　组合在一起　

    cv2.imwrite("matched.png", matched)

    win = cv2.namedWindow('test win2', flags=0)

    cv2.imshow('test win2', matched)
    cv2.waitKey();
    cv2.destroyAllWindows()
else:
    print( "Not enough matches are found - ".format(len(good),MIN_MATCH_COUNT))