# -*- coding: utf-8 -*-
"""
Created on Mon May  7 14:00:59 2018

@author: Windows
"""

import numpy as np
from skimage import data
import matplotlib.pyplot as plt

kp1 = np.array([[2.996591806411743, 27.98287010192871],[28.32323,43.23232],[2.4,4.56]])
kp5 = np.array([[2.996591806411743, 27.98287010192871]])
kp2 = np.array(kp1)
a = len(kp2)
b = len(kp1)
#print (kp1)
#print (kp2[0,0],kp2[1,0],kp1[1,0],kp1[0,0],kp2[:,1],kp1[:,1])
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
#数组的降维度 
mat=np.array([1, 2, 3, 4, 5, 6])
arr=kp1.flatten()
print (arr)
n, bins, patches = plt.hist(mat, bins=256, normed=1,edgecolor='None',facecolor='red')  
plt.show()

#概率分布直方图
#高斯分布
#均值为0
mean = 0
#标准差为1，反应数据集中还是分散的值
sigma = 1
x=mean+sigma*mat
fig,(ax0,ax1) = plt.subplots(nrows=2,figsize=(9,6))
#第二个参数是柱子宽一些还是窄一些，越大越窄越密
ax0.hist(x,40,normed=1,histtype='bar',facecolor='yellowgreen',alpha=0.75)
##pdf概率分布图，一万个数落在某个区间内的数有多少个
ax0.set_title('pdf')
ax1.hist(x,20,normed=1,histtype='bar',facecolor='pink',alpha=0.75,cumulative=True,rwidth=0.8)
#cdf累计概率函数，cumulative累计。比如需要统计小于5的数的概率
ax1.set_title("cdf")
fig.subplots_adjust(hspace=0.4)
plt.show()
