# -*- coding: utf-8 -*-
"""
Created on Mon May  7 19:41:10 2018
图像的分块搜索
@author: Windows
"""
import cv2  
import numpy as np
import math  
import string  
import os

dstPath = r'D:\code\tools\1'
n = 10
ratio = 1/n
 
imgname1 = "003-a.jpg"
imgname2 = "003-b.jpg"

img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)
    
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kpts2, descs2 = sift.detectAndCompute(gray2, None)
    
height = img1.shape[0]  
width = img1.shape[1]  
#cv2.imshow(imgPath, img)  
pHeight = int(ratio*height)  
pHeightInterval = int((height-pHeight)/(n-1))  
MIN_MATCH_COUNT = 2
#    imgname2 = "003-b.jpg"
#    img2 = cv2.imread(imgname2) 
    #print 'pHeight: %d\n' %pHeight   
    #print 'pHeightInterval: %d\n' %pHeightInterval  
      
pWidth = int(ratio*width)  
pWidthInterval = int((width-pWidth)/(n-1))  
      
    #print 'pWidth: %d\n' %pWidth   
    #print 'pWidthInterval: %d\n' %pWidthInterval  
      
cnt = 1  
for i in range(n):  
    for j in range(n):  
        x = pWidthInterval * i
        y = pHeightInterval * j
        print ('x: %d\n' %x)
        print ('y: %d\n' %y)
        print (y+pHeight)
        print (x+pWidth)
        patch = img1[y:y+pHeight, x:x+pWidth]
        gray1 = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            
        matcher = cv2.FlannBasedMatcher(dict(algorithm= 1, trees = 5), {})
        kpts1, descs1 = sift.detectAndCompute(gray1, None)
            
        matches = matcher.knnMatch(descs1, descs2, 2)
        matches = sorted(matches, key = lambda x:x[0].distance)
        good = [m1 for (m1, m2) in matches if m1.distance < 7 * m2.distance]
           
#        if len(good) < MIN_MATCH_COUNT:
            #cv2.imwrite(dstPath+'_%d' %cnt+'.jpg', patch);  
        cnt += 1
#            cv2.rectangle(img1, (x, y), (x+pWidth, y+pHeight), (0, 0, 255), 3)
matched = np.hstack((img1, img2))
cv2.imwrite("matched.png", matched)
win = cv2.namedWindow('test win', flags=0)
cv2.imshow('test win', matched)
cv2.waitKey()   
cv2.destroyAllWindows()
