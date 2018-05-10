# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:40:42 2018

@author: Windows
"""

import PIL.Image
import PIL.ImageStat

im = PIL.Image.open('d:/dog.jpg')
mean = PIL.ImageStat.Stat(im).mean
print(mean)

# [175.57884272590155, 213.62093788564377, 219.63180447003975]