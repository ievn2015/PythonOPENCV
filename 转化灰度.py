# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:46:32 2018

@author: Windows
"""

import PIL.Image

im = PIL.Image.open('d:/dog.jpg')
im = im.convert('F')