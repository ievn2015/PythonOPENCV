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

imgname1 = "002-a.jpg"
imgname2 = "002-b.jpg"

img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
matcher = cv2.FlannBasedMatcher(dict(algorithm= 1, trees = 5), {})
kpts1, descs1 = sift.detectAndCompute(gray1, None)
kpts2, descs2 = sift.detectAndCompute(gray2, None)

a = (kpts1[1].pt)
cv2.circle(img1, a, 20, (255, 255, 255), -1)
b = len(kpts1)
c = len(kpts2)
k = 0
n = 0
kp1 = []
kp2 = []
for k in range(0,b):
    kp1.append(kpts1[k].pt)
print (len(kp1))
for n in range(0,c):
    kp2.append(kpts2[n].pt)
print (len(kp2))
kp1 = np.float32(kp1)
kp2 = np.float32(kp2)
matches = matcher.knnMatch(descs1, descs2, 2)
matches = sorted(matches, key = lambda x:x[0].distance)
print (len(matches))
good = [m1 for (m1, m2) in matches if m1.distance < 0.999 * m2.distance]
src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.float32([kpts2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)

ret1 = [ i for i in kp1 if i not in src_pts ]
ret1 = np.float32(ret1)
#cv2.circle(img1, ret1[1], 20, (255, 255, 255), -1)
#print (ret1[1])
ret2 = [ i for i in kp2 if i not in dst_pts ]
ret2 = np.float32(ret2)
print (len(ret1),len(ret2))
cv2.polylines(img1,[np.int32(ret1)],True,(0,255,0),3, cv2.LINE_AA)
cv2.polylines(img2,[np.int32(ret2)],True,(0,0,255),3, cv2.LINE_AA)
matched = np.hstack((img1, img2))
cv2.imwrite("matched.png", matched)
win = cv2.namedWindow('test win2', flags=0)
cv2.imshow('test win2', matched)
cv2.waitKey();
cv2.destroyAllWindows()

