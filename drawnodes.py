import math
import random
import cv2

def draw(p00, p01, p02, p03, p04, p05, p10, p11, p12, p13, p14, p15, canvas1, canvas2): #分别依次输入两组三角形三个点坐标,两张图,并画出三角形
#COLOUR1 = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
  s = []
  d = []
  s.append(p00)
  s.append(p01)
  s.append(p02)
  s.append(p03)
  s.append(p04)
  s.append(p05)
  d.append(p10)
  d.append(p11)
  d.append(p12)
  d.append(p13)
  d.append(p14)
  d.append(p15)
  spts = []
  dpts = []
  for i in range(0,6):
        COLOUR = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        t_s = s[i] #此处常出现下标越界等问题
        t_d = d[i]
        x = int(t_s[0])
        y = int(t_s[1])
        x_1 = int(t_d[0])
        y_1 = int(t_d[1])
        spts.append((x, y))
        dpts.append((x_1, y_1))
        #分别在两张灰度图上画出对应彩色的特征点
        #print(i)
        cv2.circle(canvas1,spts[i], 20, COLOUR, -1)
        cv2.circle(canvas2,dpts[i], 20, COLOUR, -1)
        #在图上写字
        #font=cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(canvas1,'node',spts[i], font, 1,(255,255,255),2)

        #cv2.rectangle(canvas1, (x-5,y-5), (x+5, y+5),COLOUR, -1)
        #cv2.rectangle(canvas2, (x_1-5,y_1-5), (x_1+5, y_1+5),COLOUR, -1)
  #print("the difference of two cosine is", fabs1, fabs2, fabs3)
  #print (spts)
  #fabs2 = math.fabs(fabs2) #调用函数计算绝对值

  #if fabs1< 0.1 and fabs2< 0.1:
  #if fabs1 < 0.0001 and fabs2 < 0.0001 and fabs3 < 0.0001:
    #return True
  #else:
    #return False

def drawnodes(p00, canvas1): #分别依次输入两组三角形三个点坐标,两张图,并画出三角形
#COLOUR1 = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
  a = len(p00)
  COLOUR = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
  for i in range(0,a):
        t_s = p00[i] #此处常出现下标越界等问题
        x = int(t_s[0])
        y = int(t_s[1])
        #分别在两张灰度图上画出对应彩色的特征点
        #print(i)
        cv2.circle(canvas1, (x, y), 5, COLOUR, -1)