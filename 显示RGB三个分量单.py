# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 20:21:05 2018

@author: Windows
"""

import PIL.Image

im = PIL.Image.open('d:/dog.jpg')
r, g, b = im.split()

r.show()
g.show()
b.show()
#每个分量单独拿出来都是一个 [333, 250, 1] 的矩阵