# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 22:01:42 2018

@author: Windows
"""

import cv2
import numpy as np

"""
使用Sift特征点检测和匹配查找场景中特定物体。
"""

MIN_MATCH_COUNT = 4  # 至少需要4个健壮特征点

imgname1 = "009-a.jpg"
imgname2 = "009-b.jpg"

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
knnmatch先匹配，和最近无关
knnMatch是一种蛮力匹配，基本原理是将待匹配图片的sift等特征与目标图片中的全部sift特征一对n的全量遍历，
找出相似度最高的前k个。当待匹配的图片增多时，需要的计算量太大
"""
matches = matcher.knnMatch(descs1, descs2, 2)
matches = sorted(matches, key = lambda x:x[0].distance)
print (matches)
