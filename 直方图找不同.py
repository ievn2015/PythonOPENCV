# -*- coding: utf-8 -*-
"""
Created on Mon May  7 10:08:57 2018

@author: Windows
#import cv2    
import numpy as np
    
image = cv2.imread("001-a.jpg", 0)    
hist = np.float32( cv2.calcHist([image], [0], None, [256], [0.0,255.0]) )
print (hist)
"""
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread('007-a.jpg')
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()
