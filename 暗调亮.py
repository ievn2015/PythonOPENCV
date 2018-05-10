# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 12:07:04 2018

@author: Windows
"""

import PIL.Image
import scipy.misc
import numpy as np


def convert_3d(r):
    s = np.empty(r.shape, dtype=np.uint8)
    for j in range(r.shape[0]):
        for i in range(r.shape[1]):
            s[j][i] = (r[j][i] / 255) ** 0.67 * 255
    return s


im = PIL.Image.open('d:/img/jp.jpg')
im_mat = scipy.misc.fromimage(im)
im_converted_mat = convert_3d(im_mat)
im_converted = PIL.Image.fromarray(im_converted_mat)
im_converted.show()