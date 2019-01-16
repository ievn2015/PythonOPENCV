# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 08:41:04 2019

@author: IEVN
"""
import cv2
import numpy as np
import random
import SimiTri
import drawnodes
"""
使用Sift特征点检测和匹配查找场景中特定物体。
"""

MIN_MATCH_COUNT = 4  # 至少需要4个健壮特征点
# (1) 准备测试文件和数据
img1 = cv2.imread("003-a.jpg")
img2 = cv2.imread("003-b.jpg")
h, w, ch = img1.shape #得到图像1的高/宽/通道数（3）
j,k,l=img2.shape
#此处要加上判断两张图片尺寸是否一样，如果尺寸不一样会报错
#新建两张图
img3 = img4 = np.zeros([j, k, l], img2.dtype)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 转灰度
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 转灰度
#为了能在灰度图上画彩色标记
cv2.imwrite("gray1.jpg", gray1)
cv2.imwrite("gray2.jpg", gray2)
# (2) 建立SIFT 目标
sift = cv2.xfeatures2d.SIFT_create()
#建立基于Flann的匹配器
matcher = cv2.FlannBasedMatcher(dict(algorithm= 1, trees = 5), {})
#检测和得到灰度图的特征点和描述子
kpts1, descs1 = sift.detectAndCompute(gray1, None)
kpts2, descs2 = sift.detectAndCompute(gray2, None)
#用特征点描述对特征点进行knn匹配
matches = matcher.knnMatch(descs1, descs2, 2)
#对matches匹配的距离进行排序
matches = sorted(matches, key = lambda x:x[0].distance)
#取其中距离接近的
good = [m1 for (m1, m2) in matches if m1.distance < 0.5 * m2.distance]
#输出总匹配数和筛选的匹配数
print("The Value of all_matches & good_matches is")
print(len(matches), len(good))
#copy image2gray to canvas
canvas1 = cv2.imread("gray1.jpg")
canvas2 = cv2.imread("gray2.jpg")
if len(good) > MIN_MATCH_COUNT:
    #src_pts, dst_pts save the 特征点坐标,计算单应性矩阵的时候需要是numpy数组,用s——list不报错
    src_pts = list([ kpts1[m.queryIdx].pt for m in good ])
    #src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ])，为了不报错用了list,用了list不能用算单应性矩阵
    dst_pts = list([ kpts2[m.trainIdx].pt for m in good ])

    #测试随机,建立一个随机数组ran，画出特征点，成功。
    ran = []
    for i in range(0,6):
        ran.append(random.randint(0,len(src_pts)-1))
    drawnodes.draw(src_pts[ran[0]], src_pts[ran[1]], src_pts[ran[2]], src_pts[ran[3]], src_pts[ran[4]], src_pts[ran[5]], dst_pts[ran[0]], dst_pts[ran[1]], dst_pts[ran[2]], dst_pts[ran[3]], dst_pts[ran[4]], dst_pts[ran[5]], canvas1, canvas2)



    cv2.imwrite("draw1.jpg", canvas1) #保存图一画上特征点的结果
    CompareImage = np.hstack((canvas1,canvas2)) #两张图的结果拼成一张图
    cv2.imwrite("CompareImage91.jpg", CompareImage) 
    win = cv2.namedWindow('draw1', flags=0) #先新建一个窗口
    cv2.imshow('draw1', canvas1) #显示图片
    win = cv2.namedWindow('CompareImage', flags=0)
    cv2.imshow('CompareImage', CompareImage)


else:
    print("Not enough matches are found")
#遍历good中所有的特征点

cv2.waitKey()
cv2.destroyAllWindows()

