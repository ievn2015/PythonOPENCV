# -*- coding: utf-8 -*-
"""
Created on Thu May  3 11:51:12 2018

@author: Windows
"""

from PIL import Image  
from pylab import *  
  
im = array(Image.open('test.jpg'))  
imshow(im)  
x =ginput(3)  
print 'you clicked:',x  
show()  