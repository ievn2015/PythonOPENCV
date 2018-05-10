# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 20:15:09 2018

@author: Windows
"""

import scipy.misc

mat = scipy.misc.imread('d:/dog.jpg')
print(mat.shape)
# （333， 250， 3）表示333列，250行，3个色彩分量