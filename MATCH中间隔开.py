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

"""
使用Sift特征点检测和匹配查找场景中特定物体。
"""

MIN_MATCH_COUNT = 4  # 至少需要4个健壮特征点

imgname1 = "003-a.jpg"
imgname2 = "003-b.jpg"

# (1) 准备测试文件和数据
img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 转灰度
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 转灰度
H,W= img1.shape[:2]
# (2) 建立SIFT 目标
sift = cv2.xfeatures2d.SIFT_create()

matcher = cv2.FlannBasedMatcher(dict(algorithm= 1, trees = 5), {})

# (4) 提取关键点，计算关键点描述子descriptors
kpts1, descs1 = sift.detectAndCompute(gray1, None)
kpts2, descs2 = sift.detectAndCompute(gray2, None)

matches = matcher.knnMatch(descs1, descs2, 2)
# 按照距离排序
matches = sorted(matches, key = lambda x:x[0].distance)

# (6) Ratio test, to get good matches，检验比率.距离越小匹配越准确，一般选择0.7，因为图像中相似的很多，我选了0.23
good = [m1 for (m1, m2) in matches if m1.distance < 0.5* m2.distance]

canvas = img2.copy()
print(len(good),len(kpts1),good[0],kpts1[good[0].queryIdx].pt)

if len(good) > MIN_MATCH_COUNT:

    src_pts = np.int32([ kpts1[m.queryIdx].pt for m in good ])
    dst_pts = np.int32([ kpts2[m.trainIdx].pt for m in good ])
    #PT1 = src_pts[0]
    #PT2 = dst_pts[0]
    #x1 = PT1[0]
    #y1 = PT1[1]
    #x2 = PT2[0]
    #y2 = PT2[1]
    #print(x1)
    #print(src_pts,src_pts[0])
    #cv2.line(canvas,(0,0),(300,300),(0,255,0),2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    h,w = img1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

else:
    print( "Not enough matches are found - ".format(len(good),MIN_MATCH_COUNT))


h,w = img1.shape[:2]
height = h
width = 100
image = np.zeros((height, width, 3), dtype=np.uint8)
res = np.hstack((img1,image))

matched = cv2.drawMatches(res,kpts1,canvas,kpts2,good,None,(0,255,0))
res = np.hstack((res,canvas))
a = len(good)
for i in range(0,a-1):
  PT1 = src_pts[i]
  PT2 = dst_pts[i]
  x1 = PT1[0]
  y1 = PT1[1]
  x2 = PT2[0]
  y2 = PT2[1]
  if x1 < w/2 and y1 < h/2 :
    cv2.circle(res, (x1,y1), 30, (0, 255, 0), 2)
    cv2.circle(res, (x2+w+100,y2), 30, (0, 255, 0), 2)
    #cv2.line(res,(x1,y1),(x2+w+100,y2),(0,255,0),2)
  elif x1 > w/2 and y1 < h/2 :
    cv2.circle(res, (x1,y1), 30, (255, 0,0), 2)
    cv2.circle(res, (x2+w+100,y2), 30, (255, 0,0), 2)
    #cv2.line(res,(x1,y1),(x2+w+100,y2),(255,0,0),2)
  elif x1 < w/2 and y1 > h/2 :
    cv2.circle(res, (x1,y1), 30, (0, 0,255), 2)
    cv2.circle(res, (x2+w+100,y2), 30, (0,0, 255), 2)
    #cv2.line(res,(x1,y1),(x2+w+100,y2),(0,0,255),2)
  else:
    cv2.circle(res, (x1,y1), 30, (255, 255, 255), 2)
    cv2.circle(res, (x2+w+100,y2), 30, (255, 255,255), 2)
    #cv2.line(res,(x1,y1),(x2+w+100,y2),(255,255,255),2)
#cv2.line(res,(x1,y1),(x2+w+100,y2),(0,255,0),2)

pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, M)
perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
found = cv2.warpPerspective(img2,perspectiveM,(w,h)) #warp变形

cv2.imwrite("matched.png", matched)
cv2.imwrite("found.png", found)

win = cv2.namedWindow('test win', flags=0)
win = cv2.namedWindow('test win2', flags=0)
cv2.imshow("test win", res)
cv2.imshow('test win2', matched)
cv2.waitKey();cv2.destroyAllWindows()