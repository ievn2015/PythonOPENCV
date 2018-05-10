# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 23:36:11 2018

@author: Windows
"""

import cv2 as cv
from matplotlib import pyplot as plt

queryImage=cv.imread("s1.jpg",0)
trainingImage=cv.imread("s2.jpg",0)#读取要匹配的灰度照片
sift=cv.xfeatures2d.SIFT_create()#创建sift检测器
kp1, des1 = sift.detectAndCompute(queryImage,None)
kp2, des2 = sift.detectAndCompute(trainingImage,None)
#设置Flannde参数
FLANN_INDEX_KDTREE=0
indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
searchParams= dict(checks=50)
flann=cv.FlannBasedMatcher(indexParams,searchParams)
matches=flann.knnMatch(des1,des2,k=2)
#设置好初始匹配值
matchesMask=[[0,0] for i in range (len(matches))]
for i, (m,n) in enumerate(matches):
    if m.distance< 0.7*n.distance: #舍弃小于0.7的匹配结果
   matchesMask[i]=[1,0]

drawParams=dict(matchColor=(0,0,255),singlePointColor=(255,0,0),matchesMask=matchesMask,flags=0) #给特征点和匹配的线定义颜色
resultimage=cv.drawMatchesKnn(queryImage,kp1,trainingImage,kp2,matches,None,**drawParams) #画出匹配的结果
plt.imshow(resultimage,),plt.show()