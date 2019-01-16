import math
import random
import cv2
#def SimiTri([x0,y0],[x1,y1],[x2,y2],[x3,y3],[x4,y4],[x5,y5]):
def SimiTri(x0,y0,x1,y1,x2,y2,x3,y3,x4,y4,x5,y5): #输入三个点横纵坐标6个值
  a1=math.sqrt((x2-x0)*(x2-x0)+(y2-y0)*(y2-y0))
  b1=math.sqrt((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0))
  c1=math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
  a2=math.sqrt((x5-x3)*(x5-x3)+(y5-y3)*(y5-y3))
  b2=math.sqrt((x4-x3)*(x4-x3)+(y4-y3)*(y4-y3))
  c2=math.sqrt((x4-x5)*(x4-x5)+(y4-y5)*(y4-y5))

  cos0 = (a1*a1-b1*b1-c1*c1)/(-2*b1*c1)
  cos1 = (a2*a2-b2*b2-c2*c2)/(-2*b2*c2)
  #Angel=math.degrees(math.acos((a1*a1-b1*b1-c1*c1)/(-2*b1*c1)))
  
  fabs1 = b1/c1 - b2/c2
  fabs1 = math.fabs(fabs1)
  fabs2 = cos0 - cos1
  fabs2 = math.fabs(fabs2)

  if fabs1< 0.1 and fabs2< 0.1:
    return True
  else:
    return False

def SimiTri1(p00, p01, p02, p10, p11, p12): #分别依次输入两组三角形三个点坐标
  x0 = p00[0]
  y0 = p00[1]
  x1 = p01[0]
  y1 = p01[1]
  x2 = p02[0]
  y2 = p02[1]
  x3 = p10[0]
  y3 = p10[1]
  x4 = p11[0]
  y4 = p11[1]
  x5 = p12[0]
  y5 = p12[1]  
  a1=math.sqrt((x2-x0)*(x2-x0)+(y2-y0)*(y2-y0)) #1号三角形02边长a1
  b1=math.sqrt((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)) #1号三角形01边长b1
  c1=math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)) #1号三角形21边长c1
  a2=math.sqrt((x5-x3)*(x5-x3)+(y5-y3)*(y5-y3)) #2号三角形对应02边长a2
  b2=math.sqrt((x4-x3)*(x4-x3)+(y4-y3)*(y4-y3)) #2号三角形对应01边长b2
  c2=math.sqrt((x4-x5)*(x4-x5)+(y4-y5)*(y4-y5)) #2号三角形对应21边长c2
  #分别计算第一组角的两个三角函数
  cos01 = (a1*a1-b1*b1-c1*c1)/(-2*b1*c1)
  cos11 = (a2*a2-b2*b2-c2*c2)/(-2*b2*c2)
  #Angel=math.degrees(math.acos((a1*a1-b1*b1-c1*c1)/(-2*b1*c1))) # 调用函数计算反三角函数
  cos02 = (-a1*a1+b1*b1-c1*c1)/(-2*a1*c1)
  cos12 = (-a2*a2+b2*b2-c2*c2)/(-2*a2*c2)
  cos03 = (-a1*a1-b1*b1+c1*c1)/(-2*b1*a1)
  cos13 = (-a2*a2-b2*b2+c2*c2)/(-2*b2*a2)
  #fabs1 = b1/c1 -b2/c2
  #fabs1 = math.fabs(fabs1)
  fabs1 = ((cos01 - cos11)*(cos01 - cos11)) / ((cos01 + cos11)*(cos01 + cos11)) 
  fabs2 = ((cos02 - cos12)*(cos02 - cos12)) / ((cos02 + cos12)*(cos02 + cos12)) 
  fabs3 = ((cos03 - cos13)*(cos03 - cos13)) / ((cos03 + cos13)*(cos03 + cos13)) 

  #print("the difference of two cosine is", fabs1, fabs2, fabs3)
  #fabs2 = math.fabs(fabs2) #调用函数计算绝对值
  match_v1 = fabs1 + fabs2 + fabs3
  return match_v1

def SimiTri2(p00, p01, p02, p10, p11, p12, canvas1, canvas2): #分别依次输入两组三角形三个点坐标,两张图,并画出三角形
  x0 = p00[0]
  y0 = p00[1]
  x1 = p01[0]
  y1 = p01[1]
  x2 = p02[0]
  y2 = p02[1]
  x3 = p10[0]
  y3 = p10[1]
  x4 = p11[0]
  y4 = p11[1]
  x5 = p12[0]
  y5 = p12[1]  
  a1=math.sqrt((x2-x0)*(x2-x0)+(y2-y0)*(y2-y0)) #1号三角形02边长a1
  b1=math.sqrt((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)) #1号三角形01边长b1
  c1=math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)) #1号三角形21边长c1
  a2=math.sqrt((x5-x3)*(x5-x3)+(y5-y3)*(y5-y3)) #2号三角形对应02边长a2
  b2=math.sqrt((x4-x3)*(x4-x3)+(y4-y3)*(y4-y3)) #2号三角形对应01边长b2
  c2=math.sqrt((x4-x5)*(x4-x5)+(y4-y5)*(y4-y5)) #2号三角形对应21边长c2
  #分别计算第一组角的两个三角函数
  cos01 = (a1*a1-b1*b1-c1*c1)/(-2*b1*c1)
  cos11 = (a2*a2-b2*b2-c2*c2)/(-2*b2*c2)
  #Angel=math.degrees(math.acos((a1*a1-b1*b1-c1*c1)/(-2*b1*c1))) # 调用函数计算反三角函数
  cos02 = (-a1*a1+b1*b1-c1*c1)/(-2*a1*c1)
  cos12 = (-a2*a2+b2*b2-c2*c2)/(-2*a2*c2)
  cos03 = (-a1*a1-b1*b1+c1*c1)/(-2*b1*a1)
  cos13 = (-a2*a2-b2*b2+c2*c2)/(-2*b2*a2)
  #fabs1 = b1/c1 -b2/c2
  #fabs1 = math.fabs(fabs1)
  fabs1 = ((cos01 - cos11)*(cos01 - cos11)) / ((cos01 + cos11)*(cos01 + cos11)) 
  fabs2 = ((cos02 - cos12)*(cos02 - cos12)) / ((cos02 + cos12)*(cos02 + cos12)) 
  fabs3 = ((cos03 - cos13)*(cos03 - cos13)) / ((cos03 + cos13)*(cos03 + cos13)) 
  #COLOUR1 = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
  s = []
  d = []
  s.append(p00)
  s.append(p01)
  s.append(p02)
  d.append(p10)
  d.append(p11)
  d.append(p12)
  for i in range(0,3):
        COLOUR = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        t_s = s[i] #此处常出现下标越界等问题
        t_d = d[i]
        x = int(t_s[0])
        y = int(t_s[1])
        x_1 = int(t_d[0])
        y_1 = int(t_d[1])
        #分别在两张灰度图上画出对应彩色的特征点
        cv2.circle(canvas1,(x,y), 30, COLOUR, -1)
        cv2.circle(canvas2,(x_1,y_1), 30, COLOUR, -1)
        #cv2.rectangle(canvas1, (x-5,y-5), (x+5, y+5),COLOUR, -1)
        #cv2.rectangle(canvas2, (x_1-5,y_1-5), (x_1+5, y_1+5),COLOUR, -1)
  print("the difference of two cosine is", fabs1, fabs2, fabs3)
  #fabs2 = math.fabs(fabs2) #调用函数计算绝对值

  #if fabs1< 0.1 and fabs2< 0.1:
  if fabs1 < 0.0001 and fabs2 < 0.0001 and fabs3 < 0.0001:
    return True
  else:
    return False

def test(p00, p01, p02, p10, p11, p12): #分别依次输入两组三角形三个点坐标,两张图,多测试数据找到阈值
  x0 = p00[0]
  y0 = p00[1]
  x1 = p01[0]
  y1 = p01[1]
  x2 = p02[0]
  y2 = p02[1]
  x3 = p10[0]
  y3 = p10[1]
  x4 = p11[0]
  y4 = p11[1]
  x5 = p12[0]
  y5 = p12[1]  
  a1=math.sqrt((x2-x0)*(x2-x0)+(y2-y0)*(y2-y0)) #1号三角形02边长a1
  b1=math.sqrt((x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)) #1号三角形01边长b1
  c1=math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)) #1号三角形21边长c1
  a2=math.sqrt((x5-x3)*(x5-x3)+(y5-y3)*(y5-y3)) #2号三角形对应02边长a2
  b2=math.sqrt((x4-x3)*(x4-x3)+(y4-y3)*(y4-y3)) #2号三角形对应01边长b2
  c2=math.sqrt((x4-x5)*(x4-x5)+(y4-y5)*(y4-y5)) #2号三角形对应21边长c2
  #分别计算第一组角的两个三角函数
  cos01 = (a1*a1-b1*b1-c1*c1)/(-2*b1*c1)
  cos11 = (a2*a2-b2*b2-c2*c2)/(-2*b2*c2)
  #Angel=math.degrees(math.acos((a1*a1-b1*b1-c1*c1)/(-2*b1*c1))) # 调用函数计算反三角函数
  cos02 = (-a1*a1+b1*b1-c1*c1)/(-2*a1*c1)
  cos12 = (-a2*a2+b2*b2-c2*c2)/(-2*a2*c2)
  cos03 = (-a1*a1-b1*b1+c1*c1)/(-2*b1*a1)
  cos13 = (-a2*a2-b2*b2+c2*c2)/(-2*b2*a2)
  #fabs1 = b1/c1 -b2/c2
  #fabs1 = math.fabs(fabs1)
  fabs1 = ((cos01 - cos11)*(cos01 - cos11)) / ((cos01 + cos11)*(cos01 + cos11)) 
  fabs2 = ((cos02 - cos12)*(cos02 - cos12)) / ((cos02 + cos12)*(cos02 + cos12)) 
  fabs3 = ((cos03 - cos13)*(cos03 - cos13)) / ((cos03 + cos13)*(cos03 + cos13)) 
  #COLOUR1 = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
  s = []
  d = []
  s.append(p00)
  s.append(p01)
  s.append(p02)
  d.append(p10)
  d.append(p11)
  d.append(p12)
  #for i in range(0,3):
        #COLOUR = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        #t_s = s[i] #此处常出现下标越界等问题
        #t_d = d[i]
        #x = int(t_s[0])
        #y = int(t_s[1])
        #x_1 = int(t_d[0])
        #y_1 = int(t_d[1])
        #分别在两张灰度图上画出对应彩色的特征点
        #cv2.circle(canvas1,(x,y), 30, COLOUR, -1) #画圆
        #cv2.circle(canvas2,(x_1,y_1), 30, COLOUR, -1)
        #cv2.rectangle(canvas1, (x-5,y-5), (x+5, y+5),COLOUR, -1) #画方
        #cv2.rectangle(canvas2, (x_1-5,y_1-5), (x_1+5, y_1+5),COLOUR, -1)
  #print("the difference of two cosine is", fabs1, fabs2, fabs3) #打印三个组角余弦值的差
  #fabs2 = math.fabs(fabs2) #调用函数计算绝对值
  #print("Sum of 3 results is", fabs1+fabs2+fabs3) #打印三组余弦值差的和
  #if fabs1< 0.1 and fabs2< 0.1:
  match_3 = (fabs1 + fabs2 + fabs3)
  print(match_3)
  """
  #之前的设计是一个判断函数，现在改成返回每个三角形的相似度，越小越相似
  if match_3 < 0.0003:
    return True
  else:
    return False
  """
  return match_3

def fenlei_backup(src_pts, dst_pts): #创建特征点分类函数
  a = len(src_pts) 
  b = a // 3
  print("The number of points and devided by 3 is")
  print(a, b)
  match_set = []
  for i in range(0, b):
    pti0 = 3*i
    pti1 = 3*i + 1
    pti2 = 3*i + 2
    """
    pt00 = src_pts[pti0]
    pt01 = src_pts[pti1]
    pt02 = src_pts[pti2]
    pt10 = dst_pts[pti0]
    pt11 = dst_pts[pti1]
    pt12 = dst_pts[pti2]
    """
    #print(pti0, pti1, pti2)
    #print(pt00, pt01, pt02, pt10, pt11, pt12)
    match_value = SimiTri1(src_pts[pti0], src_pts[pti1], src_pts[pti2], dst_pts[pti0], dst_pts[pti1], dst_pts[pti2])
    match_set.append(match_value)
  #print(match_set)
  #matche——min返回下标
  match_min = match_set.index(min(match_set))
  print(match_min, match_set[match_min])
  mm0 = 3*match_min
  mm1 = 3*match_min + 1
  mm2 = 3*match_min + 2
  match_value_test = SimiTri1(src_pts[mm0], src_pts[mm1], src_pts[mm2], dst_pts[mm0], dst_pts[mm1], dst_pts[mm2])
  #检测下标的正确性
  #print(match_set)
  #return match_set
  print("test index")
  print(match_value_test)

def fenlei_backup2(src_pts, dst_pts, nsp, ndp): #创建特征点分类函数
  a = len(src_pts) 
  b = a // 3
  print("The number of points and devided by 3 is")
  print(b)
  match_set = []
  for i in range(0, b):
    pti0 = 3*i
    pti1 = 3*i + 1
    pti2 = 3*i + 2
    match_value = SimiTri1(src_pts[pti0], src_pts[pti1], src_pts[pti2], dst_pts[pti0], dst_pts[pti1], dst_pts[pti2])
    match_set.append(match_value)
  #matche——min返回下标
  match_min = match_set.index(min(match_set))
  #print(match_min, match_set[match_min])
  mm0 = 3*match_min
  mm1 = 3*match_min + 1
  mm2 = 3*match_min + 2
  """
  match_value_test = SimiTri1(src_pts[mm0], src_pts[mm1], src_pts[mm2], dst_pts[mm0], dst_pts[mm1], dst_pts[mm2])
  #检测下标的正确性
  #print(match_set)
  #return match_set
  print("test index")
  print(match_value_test)
  """
  nsp.append(src_pts[mm0])
  nsp.append(src_pts[mm1])
  nsp.append(src_pts[mm2])
  #移除三个特征点：

  



def rmsame(src_pts, dst_pts):
  a = len(src_pts) - 2
  #print(a)
  s = []
  for i in range(0,a):
    if (src_pts[i][0] == src_pts[i+1][0]) or (src_pts[i][0] == src_pts[i+2][0]) : #排除三角形中有相似点的可能
      s.append(i) #检测并保存相同的索引
  b = len(s)
  for j in range(0,b):
    src_pts.pop(s[j]-j) #根据索引删除列表中相同的项，每删除一个索引要向前挪一个
    dst_pts.pop(s[j]-j)
    #src_pts.remove(a[j])
    #dst_pts.remove(d[j])
  print(len(src_pts))

def testsame(src_pts, dst_pts):
  a = len(src_pts) - 1
  #print(a)
  #s = []
  for i in range(0,a):
    if dst_pts[i][0] == dst_pts[i+1][0]: #如果第一个坐标一样认为是相同的两个点，打印出来，这样可以测试有没有删除成功
      print(src_pts[i])

def fenlei(src_pts, dst_pts, nsp, ndp): #创建特征点分类函数
  a = len(src_pts) 
  b = a // 3
  print("The number of points and devided by 3 is")
  print(b)
  match_set = []
  for i in range(0, b):
    pti0 = 3*i
    pti1 = 3*i + 1
    pti2 = 3*i + 2
    match_value = SimiTri1(src_pts[pti0], src_pts[pti1], src_pts[pti2], dst_pts[pti0], dst_pts[pti1], dst_pts[pti2])
    match_set.append(match_value)
  #matche——min返回下标
  match_min = match_set.index(min(match_set))
  #print(match_min, match_set[match_min])
  mm0 = 3*match_min
  mm1 = 3*match_min + 1
  mm2 = 3*match_min + 2
  """
  match_value_test = SimiTri1(src_pts[mm0], src_pts[mm1], src_pts[mm2], dst_pts[mm0], dst_pts[mm1], dst_pts[mm2])
  #检测下标的正确性
  #print(match_set)
  #return match_set
  print("test index")
  print(match_value_test)
  """
  nsp_t = []
  ndp_t = []
  nsp_t.append(src_pts[mm0])
  nsp_t.append(src_pts[mm1])
  nsp_t.append(src_pts[mm2])
  ndp_t.append(dst_pts[mm0])
  ndp_t.append(dst_pts[mm1])
  ndp_t.append(dst_pts[mm2])
  for i in range(0,3):
    src_pts.pop(mm0) #根据索引删除列表中的项，每删除一个索引要向前挪一个
    dst_pts.pop(mm0)
  a = a - 3
  rr = []
  ra = []
  for k in range(0, a):
    if (nsp_t[0][0] == src_pts[k][0]) or (nsp_t[1][0] == src_pts[k][0]):
      rr.append(k)
    
    else:
        match_v2 = SimiTri1(nsp_t[0], nsp_t[1], src_pts[k], ndp_t[0], ndp_t[1], dst_pts[k])
        if match_v2 < 0.000001:
            rr.append(k)
            ra.append(k)
            nsp_t.append(src_pts[k])
            ndp_t.append(dst_pts[k])
  b = len(rr)
  for q in range(0,b):
    src_pts.pop(rr[q]-q) #根据索引删除列表中相同的项，每删除一个索引要向前挪一个
    dst_pts.pop(rr[q]-q)
  
  print(len(rr))
  
  nsp.extend(nsp_t)
  ndp.extend(ndp_t)
  print(len(nsp))

def rmsameall(src_pts, dst_pts):
  a = len(src_pts) - 1
  #print(a)
  s = []
  for i in range(0, a):
      for j in range(i + 1, a + 1):
        if src_pts[i][0] == src_pts[j][0] or dst_pts[i][0] == dst_pts[j][0]: 
            s.append(i)
            break
  b = len(s)#排除三角形中有相似点的可能
  for q in range(0,b):
        src_pts.pop(s[q]-q)
        dst_pts.pop(s[q]-q)
  print(len(src_pts))