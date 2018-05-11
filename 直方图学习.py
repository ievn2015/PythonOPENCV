# -*- coding: utf-8 -*-
"""
Created on Fri May 11 11:34:04 2018

@author: Windows
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import mlab
from matplotlib import rcParams
fig1 = plt.figure(2)
rects =plt.bar(left = (0.2,1),height = (1,0.5),width = 0.2,align="center",yerr=0.000001)
plt.title('Pe')
plt.show()