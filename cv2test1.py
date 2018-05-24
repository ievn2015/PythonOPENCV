# -*- coding: utf-8 -*-
"""
Created on Wed May 23 20:43:19 2018

@author: Windows
"""
import cv2
imgname1 = "003-a.jpg"
imgname2 = "003-b.jpg"
img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)

win = cv2.namedWindow('Result', flags=0)
cv2.imshow('Result', img1)
cv2.waitKey()
cv2.destroyAllWindows()