# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 17:27:08 2018

@author: SJW
"""

# /usr/bin/python3
# 2017.11.11 01:44:37 CST
# 2017.11.12 00:09:14 CST

import cv2
import numpy as np
import time
MIN_MATCH_COUNT = 4 

imgname1 = "ttt1.jpg"
imgname2 = "ttt2.jpg"

img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

matcher = cv2.FlannBasedMatcher(dict(algorithm= 1, trees = 5), {})

kpts1, descs1 = sift.detectAndCompute(gray1, None)
kpts2, descs2 = sift.detectAndCompute(gray2, None)


matches = matcher.knnMatch(descs1, descs2, 2)

matches = sorted(matches, key = lambda x:x[0].distance)

good = [m1 for (m1, m2) in matches if m1.distance < 0.5* m2.distance]
print(len(good))
canvas = img2.copy()

if len(good) > MIN_MATCH_COUNT:

    src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)




    h,w = img1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    cv2.polylines(canvas,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)

    matched = np.hstack((img1,canvas))

    perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
    found = cv2.warpPerspective(img2,perspectiveM,(w,h))
    res = np.hstack((img1,found))

    img3 = img1 - found 
    win = cv2.namedWindow('test win2', flags=0)
    win = cv2.namedWindow('test win', flags=0)
    cv2.imshow('test win', img3)
    cv2.imshow("test win2", res)
    #cnt = 0
    #cv2.imwrite('D:\Documents\项目汇报\1026\1.jpg', matched )
    #cnt += 1
    #cv2.imwrite('D:\Documents\项目汇报\1026\2.jpg', res)
    


else:
    print( "Hello, World!" )
cv2.waitKey()
cv2.destroyAllWindows()