# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:00:59 2018

@author: Windows
"""

import numpy as np

kp1 = np.array([[2.996591806411743, 27.98287010192871],[28.32323,43.23232],[2.4,4.56]])
kp5 = np.array([[2.996591806411743, 27.98287010192871]])
kp2 = np.array(kp1)
a = len(kp2)
b = len(kp1)
print (kp1)
print (kp2[0,0],kp2[1,0],kp1[1,0],kp1[0,0],kp2[:,1],kp1[:,1])
# 得到一个逻辑数组
index = kp1 >2
kp3 = np.array(index)
#print(kp3)

index2 = np.where(kp1>5)
kp4 = np.array(index2)
#print (kp4)
kp4 = np.delete(kp1,index2)
#print (kp4)
ret = [ i for i in kp1 if i not in kp5 ]
ret = np.float32(ret)
#print (ret)
ret2 = []
for j in kp1:
    if j not in kp5:
        ret2.append(j)
#print (ret2)