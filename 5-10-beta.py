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
good1 = [m1 for (m1, m2) in matches if m1.distance < 0.9 * m2.distance]

src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) 
dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
src_pts1 = np.float32([ kpts1[m.queryIdx].pt for m in good1 ]).reshape(-1,1,2) 
dst_pts1 = np.float32([ kpts2[m.trainIdx].pt for m in good1 ]).reshape(-1,1,2) 

a = len(dst_pts)
#j = 0
src_sub_a = []
dst_sub_a = []
for i in range(0,a):
    dist = np.linalg.norm(src_pts[i] - dst_pts[i])
    if dist > 150:
        src_sub_a.append(src_pts[i])
        dst_sub_a.append(dst_pts[i])
        
src_sub_a = np.float32(src_sub_a)
dst_sub_a = np.float32(dst_sub_a)
j = len(src_sub_a)
print(j)
src_sub_a = src_sub_a[0:j,0]
X = src_sub_a[:,0]
#print(X)
maxX = int(max(X))
minX = int(min(X))
#print (maxX)
Y = src_sub_a[:,1]
maxY = int(max(Y))
minY = int(min(Y))

j = len(dst_sub_a)
print(j)
dst_sub_a = dst_sub_a[0:j,0]
#print (dst_sub_a)
X1 = dst_sub_a[:,0]
maxX1 = int(max(X1))
minX1 = int(min(X1))
print (maxX1)
#print (X1)
Y1 = src_sub_a[:,1]
maxY1 = max(Y1)
minY1 = min(Y1)

if len(good1) > MIN_MATCH_COUNT:

    a = len(src_pts1)/2
    a = int (a)
    
    c = src_pts1.ravel()
    d = dst_pts1.ravel()

    canvas1 = gray1.copy()
    canvas2 = gray2.copy()

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
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    (_, cnts, _) = cv2.findContours(closed.copy(), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_NONE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    print (len(c))
    if len(c) > 150 :
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
    #dist1 = np.linalg.norm(box[0] - box[-1])
    #print (dist1)
    #if dist1 <100:
        draw_img1 = cv2.drawContours(img1.copy(), [box], -1, (255, 0, 0), 15)
    #else:
        #draw_img1 = img1.copy()
    else:
        draw_img1 = img1.copy()
    blurred = cv2.GaussianBlur(gradient2, (9, 9),0)
    (_, thresh) = cv2.threshold(blurred, 215, 0, 4)
    (_, thresh) = cv2.threshold(thresh, 40, 0, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    (_, cnts, _) = cv2.findContours(closed.copy(),
        cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    #print (c)
    if len(c) > 150 :
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))
    #dist1 = np.linalg.norm(box[0] - box[-1])
    #print (dist1)
    #if dist1 <100:
        draw_img2 = cv2.drawContours(img2.copy(), [box], -1, (255, 0, 0), 15)
    #else:
        #draw_img1 = img1.copy()
    else:
        draw_img2 = img2.copy()
    cv2.rectangle(draw_img1, (minX, minY), (maxX, maxY), (0, 0, 255), 20)
    cv2.rectangle(draw_img2, (minX1, minY1), (maxX1, maxY1), (0, 0, 255), 20)
    #cv2.polylines(draw_img1, [np.int32(src_sub_a)],True,(0,255,0),3, cv2.LINE_AA)
    #cv2.polylines(draw_img2, [np.int32(dst_sub_a)],True,(0,0,255),3, cv2.LINE_AA)
    matched = np.hstack((draw_img1, draw_img2))
    cv2.imwrite("Result7.png", matched)
    win = cv2.namedWindow('Result', flags=0)
    cv2.imshow('Result', matched)
    cv2.waitKey()
    cv2.destroyAllWindows()
else:
    print( "Can not Compare")