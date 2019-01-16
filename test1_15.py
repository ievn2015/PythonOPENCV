
import random
"""
for i in range(0,6):
    ran = []
    ran.append(random.randint(0,255))

#print(155/3, 155//3, 155%3)
c = [-10,-5,0,5,3,10,15,-20,25]
#print c.index(min(c))
#c中的最小值：min(c)
print(c.index(min(c))) #打印c中最小值的索引
"""
set0 = []
for i in range(0,6):
    a = []
    a.append(i)
    set0.append(a)
#print(set0)

a = 10
b = 10
if 2 > 1 :
    a = 1
else:
    b = 1
#print(a, b, 1)




nums = [3,-10,-5,0,5,3,3,3,10,15,5,-20,-5,25,0]
a = len(nums) - 1
#print(a)
s = []
for i in range(0, a):
    for j in range(i + 1, a + 1):
        if nums[i] == nums[j]:
            s.append(i)
b = len(s)
for q in range(0,b):
    nums.pop(s[q]-q)
print(nums)

def rmsameall(src_pts, dst_pts):
  a = len(src_pts) - 1
  #print(a)
  s = []
  for i in range(0, a):
      for j in range(i + 1, a + 1):
        if (src_pts[i][0] == src_pts[j][0] or dst_pts[i][0] == dst_pts[j][0]): 
            s.append(i)
  b = len(s)#排除三角形中有相似点的可能
  for q in range(0,b):
      src_pts.remove(s[q]-q)
      dst_pts.remove(s[q]-q)
  print(len(src_pts))