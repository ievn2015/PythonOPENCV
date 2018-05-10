# -*- coding: utf-8 -*-
"""
Created on Wed May  2 10:17:02 2018

@author: Windows
"""

# -*- coding: utf-8 -*-
from PCV.localdescriptors import sift, dsift
from PIL import Image
import pylab

dsift.process_image_dsift('empire.jpg','empire.dsift',90,40,True)
l,d = sift.read_features_from_file('empire.dsift')
im = pylab.array(Image.open('../data/empire.jpg'))
sift.plot_features(im,l,True)
pylab.show()