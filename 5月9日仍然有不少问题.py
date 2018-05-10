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
good = [m1 for (m1, m2) in matches if m1.distance < 0.4 * m2.distance]
good1 = [m1 for (m1, m2) in matches if m1.distance < 0.9 * m2.distance] #第一幅图0.4才好

src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) #获取匹配到的坐标存在np中第一张图
dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2) #获取匹配到的坐标存在np中第二张图
src_pts1 = np.float32([ kpts1[m.queryIdx].pt for m in good1 ]).reshape(-1,1,2) #获取匹配到的坐标存在np中第一张图
dst_pts1 = np.float32([ kpts2[m.trainIdx].pt for m in good1 ]).reshape(-1,1,2) #获取匹配到的坐标存在np中第二张图
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

print(len(good))
if len(good1) > MIN_MATCH_COUNT:
# 可以都用m.queryIdx
    a = len(src_pts1)/2
    a = int (a)
    print(a,len(src_pts1))
    c = src_pts1.ravel()
    d = dst_pts1.ravel()
    #c = c.reshape(2,1)
    #c = np.tuple()
    canvas1 = gray1.copy()
    canvas2 = gray2.copy()
    #print (c[0],c[1])
    for k in range(1,a):
        cv2.circle(canvas1, (c[2*k],c[2*k-1]), 80, (255, 255, 255), -1)
        cv2.circle(canvas2, (d[2*k],d[2*k-1]), 80, (255, 255, 255), -1)
    
    blurred1 = cv2.GaussianBlur(canvas1, (9, 9),0)
    blurred2 = cv2.GaussianBlur(canvas2, (9, 9),0)
    gradX1 = cv2.Sobel(blurred1, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY1 = cv2.Sobel(blurred1, ddepth=cv2.CV_32F, dx=0, dy=1)

    gradient1 = cv2.subtract(gradX1, gradY1)
    gradient1 = cv2.convertScaleAbs(gradient1)
    
    gradX2 = cv2.Sobel(blurred2, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY2 = cv2.Sobel(blurred2, ddepth=cv2.CV_32F, dx=0, dy=1)

    gradient2 = cv2.subtract(gradX2, gradY2)
    gradient2 = cv2.convertScaleAbs(gradient2)
    
    blurred = cv2.GaussianBlur(gradient1, (9, 9),0)
    (_, thresh) = cv2.threshold(blurred, 225, 0, 4)
    (_, thresh) = cv2.threshold(thresh, 30, 0, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    # 执行图像形态学, 细节直接查文档，很简单
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    (_, cnts, _) = cv2.findContours(closed.copy(), 
        cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # compute the rotated bounding box of the largest contour
    print (c)
    rect = cv2.minAreaRect(c)
    print (rect)
    box = np.int0(cv2.boxPoints(rect))
    print (box)
    draw_img1 = cv2.drawContours(img1.copy(), [box], -1, (255, 0, 0), 15)
    
    blurred = cv2.GaussianBlur(gradient2, (9, 9),0)
    (_, thresh) = cv2.threshold(blurred, 225, 0, 4)
    (_, thresh) = cv2.threshold(thresh, 30, 0, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    # 执行图像形态学, 细节直接查文档，很简单
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    (_, cnts, _) = cv2.findContours(closed.copy(), 
        cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    # compute the rotated bounding box of the largest contour
    print (c)
    rect = cv2.minAreaRect(c)
    print (rect)
    box = np.int0(cv2.boxPoints(rect))
    print (box)
    draw_img2 = cv2.drawContours(img2.copy(), [box], -1, (255, 0, 0), 15)
    
    
    
    cv2.polylines(draw_img1, [np.int32(src_sub_a)],True,(0,255,0),3, cv2.LINE_AA)
    cv2.polylines(draw_img2,[np.int32(dst_sub_a)],True,(0,0,255),3, cv2.LINE_AA)
    matched = np.hstack((draw_img1, draw_img2))
    cv2.imwrite("Result.png", matched)
    win = cv2.namedWindow('Result', flags=0)
    cv2.imshow('Result', matched)
    cv2.waitKey();
    cv2.destroyAllWindows()
else:
    print( "Not enough matches are found - ".format(len(good),MIN_MATCH_COUNT))