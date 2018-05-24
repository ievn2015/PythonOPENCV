# -*- coding: utf-8 -*-
"""
Created on Wed May 23 22:40:41 2018

@author: Windows
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
img=cv2.imread("009-a.jpg")
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#颜色转为灰度
ret,thresh=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)#可为图像设一个阈值
kernel=np.ones((3,3),np.uint8)
opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=2)#去除噪声
sure_bg=cv2.dilate(opening,kernel,iterations=3)

dist_transform=cv2.distanceTransform(opening,cv2.DIST_L2,5)#可以通过distanceTransform来获取确定的前景区域。也就是说，这是图像中最可能是前景的区域，越是远离背景区域的边界点越可能属于前景，这里用了阈值来决定那些区域是前景
ret,sure_fg=cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#这个阶段之后，所得到的前景和背景中有重合的部分该怎么办？首先需要确定这些区域，这可从sure_bg与sure_fg的集合相减得到
sure_fg=np.uint8(sure_fg)
unknown=cv2.subtract(sure_bg,sure_fg)
#现在有了这些区域，就可以设定栅栏来阻止水汇聚，这是通过connectedComponents函数完成。
ret,markers=cv2.connectedComponents(sure_fg)

markers=markers+1
#markers[unknown==255]=0
#把栅栏绘制成红色
markers=cv2.watershed(img,markers)
img[markers==-1]=[255,0,0]
cv2.imshow('Key Points', img)
cv2.waitKey()
cv2.destroyAllWindows()