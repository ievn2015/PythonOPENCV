# -*- coding: utf-8 -*-
"""
Created on Thu May  3 21:56:11 2018

@author: Windows
"""

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

sift = cv2.xfeatures2d.SIFT_create()

matcher = cv2.FlannBasedMatcher(dict(algorithm= 1, trees = 5), {})

kpts1, descs1 = sift.detectAndCompute(gray1, None,)
kpts2, descs2 = sift.detectAndCompute(gray2, None,)

matches = matcher.knnMatch(descs1, descs2, 2)

matches = sorted(matches, key = lambda x:x[0].distance)

good = [m1 for (m1, m2) in matches if m1.distance < 0.5 * m2.distance]

canvas = img2.copy()

if len(good) > MIN_MATCH_COUNT:
    # 从匹配中提取出对应点对
    # (queryIndex for the small object, trainIndex for the scene )
    src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2) #获取匹配到的坐标存在np中第一张图
    dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2) #获取匹配到的坐标存在np中第二张图
    print (src_pts)
   # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
   # print(M)
    a = len(dst_pts)
    print (a)
  #  b = len(src_pts)(两个都是一样长，但是数量会不一样)
    src_sub_copy = src_pts[:4, :2].copy() #数组切片，第一个参数是取几行，第二个参数是取几列
    dst_sub_copy = dst_pts[:4, :2].copy()
    print(src_sub_copy)
   # print (a , b)
   # print (src_pts[0],src_pts[1],dst_pts[a-1])  
    #N, mask = cv2.findHomography(src_sub_copy, dst_sub_copy, cv2.RANSAC,5.0)
    #print (N)
    # find homography matrix in cv2.RANSAC(随机抽样) using good match points

    # 掩模，用作绘制计算单应性矩阵时用到的点对
    #matchesMask2 = mask.ravel().tolist()
    # 计算图1的畸变，也就是在图2中的对应的位置。
  #  h,w = img1.shape[:2]
  #  pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
 #   dst = cv2.perspectiveTransform(pts,M)
    # 绘制边框
#    cv2.polylines(img1,[np.int32(src_sub_copy)],True,(255,0,0),3, cv2.LINE_AA)
#    cv2.polylines(canvas,[np.int32(dst_sub_copy)],True,(255,0,255),3, cv2.LINE_AA) 
  
#    src_su_copy = src_pts[5:9, :2].copy() #数组切片，第一个参数是取几行，第二个参数是取几列
#    dst_su_copy = dst_pts[5:9, :2].copy()
#    print(src_su_copy)
   # print (a , b)
#    M, mask = cv2.findHomography(src_su_copy, dst_su_copy, cv2.RANSAC,5.0)
#    print (M)
#    cv2.polylines(img1,[np.int32(src_su_copy)],True,(255,0,0),3, cv2.LINE_AA)
#    cv2.polylines(canvas,[np.int32(dst_su_copy)],True,(255,0,0),3, cv2.LINE_AA)

    # 括号内颜色，20越大越粗
    #画框需要把浮点型化成整形
    
# (8) 画线
#    draw_params = dict(matchColor = (0,0,255),
#                   singlePointColor = (0,255,0),
#                   flags = 0)
#    matched = cv2.drawMatches(img1,kpts1,canvas,kpts2,good,None,**draw_params)# 括号内为连线颜色，不写则颜色随机,可以在后面加特征点颜色，不写就颜色随机)
    matched = np.hstack((img1, canvas))   #　组合在一起　
    #cv2.imwrite("matched.png", res)    
    # (9) 从场景中提取
#    h,w = img1.shape[:2]
#    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
#    dst = cv2.perspectiveTransform(pts, M)
 #   perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
 #   found = cv2.warpPerspective(img2,perspectiveM,(w,h)) #warp变形
    
    # (10) 保存显示
    cv2.imwrite("matched.png", matched)
#    cv2.imwrite("found.png", found)
    # 先创建窗口再显示图片，直接cv2显示图片无法调整图片大小，但是这种窗口显示会导致纵横比失调
 #   win = cv2.namedWindow('test win', flags=0)
    win = cv2.namedWindow('test win2', flags=0)
  #  cv2.imshow("test win", found)
    cv2.imshow('test win2', matched)
    cv2.waitKey();
    cv2.destroyAllWindows()
else:
    print( "Not enough matches are found - ".format(len(good),MIN_MATCH_COUNT))