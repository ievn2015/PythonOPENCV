# -*- coding: utf-8 -*-
"""
Created on Tue May 22 04:19:36 2018

@author: IEVN
"""

import matplotlib.pyplot as plt
from skimage import data,color,morphology

#生成二值测试图像
img=color.rgb2gray(data.horse())
img=(img<0.5)*1

chull = morphology.convex_hull_image(img)

#绘制轮廓
fig, axes = plt.subplots(1,2,figsize=(8,8))
ax0, ax1= axes.ravel()
ax0.imshow(img,plt.cm.gray)
ax0.set_title('original image')

ax1.imshow(chull,plt.cm.gray)
ax1.set_title('convex_hull image')