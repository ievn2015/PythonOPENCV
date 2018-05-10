# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 20:47:47 2018

@author: Windows
"""

import PIL.Image

im = PIL.Image.open('d:/th.jpg')
r, g, b = im.split()
im = PIL.Image.merge('RGB', (b, g, r))
im.show()
#交换颜色