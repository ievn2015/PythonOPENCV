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

imgname1 = "s1.jpg"
imgname2 = "s2.jpg"

# (1) 准备测试文件和数据
img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 转灰度
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 转灰度


# (2) 建立SIFT 目标
sift = cv2.xfeatures2d.SIFT_create()
"""
 Fast Library for Approximate Nearest Neighbors，是FLANN的全称
(3) 建立flann匹配，（近似）最近邻开源库。实现了一系列查找算法，还包含了一种自动选取最快算法的机制。 
使用SFIT特征提取关键点
使用FLANN匹配器进行描述子向量匹配
OpenCV提供了 两种Matching方式 ：
• Brute-force matcher (cv::BFMatcher)
• Flann-based matcher (cv::FlannBasedMatcher)
Brute-force matcher就是用暴力方法找到点集一中每个descriptor在点集二中距离最近的 descriptor；
Flann-based matcher 使用快速近似最近邻搜索算法寻找。
是否可以机器学习的思想先训练一个matcher，然后匹配物体
关于以下API的调用，关键点的匹配可以采用穷举法来完成，但是这样耗费的时间太多，一
般都采用一种叫kd树的数据结构来完成搜索。搜索的内容是以目标图
像的关键点为基准，搜索与目标图像的特征点最邻近的原图像特征点
和次邻近的原图像特征点。
关键点匹配
Kd树是一个平衡二叉树
官方教程,http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
"""
matcher = cv2.FlannBasedMatcher(dict(algorithm= 1, trees = 5), {})

# (4) 提取关键点，计算关键点描述子descriptors
kpts1, descs1 = sift.detectAndCompute(gray1, None)
kpts2, descs2 = sift.detectAndCompute(gray2, None)

# (5) knnMatch匹配前2
"""
knnMatch是一种蛮力匹配，基本原理是将待匹配图片的sift等特征与目标图片中的全部sift特征一对n的全量遍历，
找出相似度最高的前k个。当待匹配的图片增多时，需要的计算量太大
"""
matches = matcher.knnMatch(descs1, descs2, 2)
# 按照距离排序
matches = sorted(matches, key = lambda x:x[0].distance)

# (6) Ratio test, to get good matches，检验比率.距离越小匹配越准确，一般选择0.7，因为图像中相似的很多，我选了0.23
good = [m1 for (m1, m2) in matches if m1.distance < 0.23 * m2.distance]

canvas = img2.copy()
"""
# (7) find homography matrix，寻找单应性矩阵
# 当有足够的健壮匹配点对（至少4个）时
单应性是一个条件，该条件表面当两幅图像中的一副出像投影畸变时，他们还能匹配。
在计算机视觉中：对单应性最感兴趣的部分只是其他意义的一个子集。
平面的单应性被定义为从一个平面到另一个平面的投影映射。
比如，一个二维平面上的点映射到摄像机成像仪上的映射就是平面单应性的例子。
"""

if len(good) > MIN_MATCH_COUNT:
    # 从匹配中提取出对应点对
    # (queryIndex for the small object, trainIndex for the scene )
    src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    # find homography matrix in cv2.RANSAC(随机抽样) using good match points
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    # 掩模，用作绘制计算单应性矩阵时用到的点对
    #matchesMask2 = mask.ravel().tolist()
    # 计算图1的畸变，也就是在图2中的对应的位置。
    h,w = img1.shape[:2]
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    # 绘制边框
    cv2.polylines(canvas,[np.int32(dst)],True,(0,255,0),20, cv2.LINE_AA) # 括号内颜色，20越大越粗
else:
    print( "Not enough matches are found - ".format(len(good),MIN_MATCH_COUNT))

## (8) drawMatches
matched = cv2.drawMatches(img1,kpts1,canvas,kpts2,good,None,(0,0,255))#括号内为连线颜色，不写则颜色随机,可以在后面加特征点颜色，不写就颜色随机)

## (9) 从场景中提取
h,w = img1.shape[:2]
pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts, M)
perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
found = cv2.warpPerspective(img2,perspectiveM,(w,h)) #warp变形

## (10) 保存显示
cv2.imwrite("matched.png", matched)
cv2.imwrite("found.png", found)
## 先创建窗口再显示图片，直接cv2显示图片无法调整图片大小，但是这种窗口显示会导致纵横比失调
win = cv2.namedWindow('test win', flags=0)
win = cv2.namedWindow('test win2', flags=0)
cv2.imshow("test win", found)
cv2.imshow('test win2', matched)
cv2.waitKey();cv2.destroyAllWindows()