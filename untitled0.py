# -*- coding: utf-8 -*-
"""
Created on Mon May  7 19:41:10 2018
图像的分块搜索
@author: Windows
"""
import cv2  
import numpy  
import math  
import string  
import os  
def split(  
    img,    #image matrix  
    ratio,  #patch_length/image_length  
    n,      #number of patches per line  
    dstPath #destination path  
    ):  
    height = img.shape[0]  
    width = img.shape[1]  
    #cv2.imshow(imgPath, img)  
    pHeight = int(ratio*height)  
    pHeightInterval = (height-pHeight)/(n-1)  
      
    #print 'pHeight: %d\n' %pHeight   
    #print 'pHeightInterval: %d\n' %pHeightInterval  
      
    pWidth = int(ratio*width)  
    pWidthInterval = (width-pWidth)/(n-1)  
      
    #print 'pWidth: %d\n' %pWidth   
    #print 'pWidthInterval: %d\n' %pWidthInterval  
      
    cnt = 1  
    for i in range(n):  
        for j in range(n):  
            x = pWidthInterval * i  
            y = pHeightInterval * j  
              
            #print 'x: %d\n' %x  
            #print 'y: %d\n' %y  
              
            patch = img[y:y+pHeight, x:x+pWidth, :]  
            cv2.imwrite(dstPath+'_%d' %cnt+'.jpg', patch);  
            cnt += 1  
            #cv2.imshow('patch',patch)  
            #cv2.waitKey(0)
