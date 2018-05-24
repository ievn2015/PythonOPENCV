# -*- coding: utf-8 -*-
"""
Created on Thu May 24 02:15:05 2018

@author: Windows
"""

    
    src_sub_a = np.int32(src_sub_a)
    dst_sub_a = np.int32(dst_sub_a)
    j = len(src_sub_a)
    src_sub_a = src_sub_a[0:j,0]
    print (src_sub_a)
    X = src_sub_a[:,0]
    maxX = max(X)
    minX = min(X)
    print (maxX)
    print (X)
    Y = src_sub_a[:,1]
    maxY = max(Y)
    minY = min(Y)
    
    j = len(dst_sub_a)
    dst_sub_a = dst_sub_a[0:j,0]
    print (dst_sub_a)
    X1 = dst_sub_a[:,0]
    maxX1 = max(X1)
    minX1 = min(X1)
    print (maxX1)
    print (X1)
    Y1 = src_sub_a[:,1]
    maxY1 = max(Y1)
    minY1 = min(Y1)
    