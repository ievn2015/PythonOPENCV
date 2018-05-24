# -*- coding: utf-8 -*-
"""
Created on Wed May 23 22:47:50 2018

@author: Windows
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

img=cv2.imread('003-b.jpg')
mask=np.zeros(img.shape[:2],np.uint8)#创建一个掩模
print(len(img.shape[:2]))
print(mask)

#创建以0填充的前景和背景模型
bgdModel=np.zeros((1,65),np.float64)
print (bgdModel)
fgdModel=np.zeros((1,65),np.float64)
print(fgdModel)
rect=(600,200,650,600)#创建矩形（左上横纵，右下横纵）
cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)#使用了指定的空模型和掩模来运行GrabCut，并且实际上是用一个矩形来初始化这个操作
#做完这些后，我们的掩模已经变成包含0~3之间的值。值为0和2的将转为0，值为1，3的将转为1.然后保存在mask2中。这样就可以用mask2过滤出所有的0值像素（理论上会完整保留所有前景像素）
mask2=np.where((mask==0)|(mask==2),0,1).astype('uint8')
img=img*mask2[:,:,np.newaxis]
win = cv2.namedWindow('Result', flags=0)
cv2.imshow('Result', img)
cv2.waitKey()
cv2.destroyAllWindows()
plt.subplot(121),plt.imshow(img)
plt.title("grabcut"),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(cv2.cvtColor(cv2.imread("1.jpg"),cv2.COLOR_BGR2RGB))
plt.title("original"),plt.xticks([]),plt.yticks([])
plt.show()

