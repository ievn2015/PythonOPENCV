import cv2
import numpy as np
import random
import compare2
import test1

"""
使用Sift特征点检测和匹配查找场景中特定物体。
"""

MIN_MATCH_COUNT = 4  # 至少需要4个健壮特征点

imgname1 = "003-a.jpg"
imgname2 = "003-b.jpg"

# (1) 准备测试文件和数据
img1 = cv2.imread(imgname1)
img2 = cv2.imread(imgname2)
h, w, ch = img1.shape
j,k,l=img2.shape
img4=np.zeros([j, k, l], img2.dtype)
img3 = np.zeros([h, w, ch], img1.dtype)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 转灰度
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  # 转灰度

# (2) 建立SIFT 目标
sift = cv2.xfeatures2d.SIFT_create()


matcher = cv2.FlannBasedMatcher(dict(algorithm= 1, trees = 5), {})

kpts1, descs1 = sift.detectAndCompute(gray1, None)
kpts2, descs2 = sift.detectAndCompute(gray2, None)

matches = matcher.knnMatch(descs1, descs2, 2)

matches = sorted(matches, key = lambda x:x[0].distance)

good = [m1 for (m1, m2) in matches if m1.distance < 0.5 * m2.distance]

canvas = img2.copy()

if len(good) > MIN_MATCH_COUNT:

    src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ])
    dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ])

    #M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    #h,w = img1.shape[:2]
    #pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    #dst = cv2.perspectiveTransform(pts,M)

else:
    print( "Not enough matches are found - ".format(len(good),MIN_MATCH_COUNT))

for i in range(0,len(src_pts)-1):
    b = src_pts[i]
    x = int(b[0])
    y = int(b[1])
    #print(x)


    cv2.rectangle(img3, (x-10,y-10), (x+10, y+10),(255,255,255), -1)
gray8 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
_a, cnts, _b= cv2.findContours(gray8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
d = sorted(cnts, key=cv2.contourArea, reverse=True)
areas = np.zeros( len(d) )
idx = 0
for cont in d : 
    areas[idx] = cv2.contourArea(cont)
    #print(areas[idx])
    idx = idx + 1

n=0
p=0
q=0
A=[]
D=[]
E=[]
M=[]
N=[]
for i in range(len(areas)):
    if areas[i] ==400:
        break
    c=d[i]
    rect = cv2.minAreaRect(c)
# rect = cv2.minAreaRect(cnts[1])
    box = np.int0(cv2.boxPoints(rect))
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    x3=int((x1+x2)/2)
    y1 = min(Ys)
    y2 = max(Ys)
    y3=int((y1+y2)/2)
    B=[]
    C=[]
    for i in range(len(src_pts)):
        U = src_pts[i]
        I = int(U[0])
        O = int(U[1])
        dist = np.linalg.norm(src_pts[i] - dst_pts[i])
        if I>=x1 and I<=x2 and O>=y1 and O<=y2 and dist<=100000:
            C.append(dst_pts[i])
            B.append(src_pts[i])
    #print(len(B))
    A.append(B)
    if len(B)>=4 :
        CL = (random.randint(0,255),random.randint(0,255),random.randint(0,255))        
        Xs1 = [i[0] for i in C]
        Ys1 = [i[1] for i in C]
        x11 = int(min(Xs1))
        x21 = int(max(Xs1))
        x31=int((x11+x21)/2)
        y11 = int(min(Ys1))
        y21 = int(max(Ys1))
        y31=int((y11+y21)/2)
        S1=(y2-y1+40)*(x2-x1+40)
        S2=(y21-y11+60)*(x21-x11+60)
        #print(S1,S2)
        if S1/S2>=0.8 and S1/S2<=1.25 :
            D.append([x3,y3])
            E.append([x31,y31])
            #for i in range(len(D)):
                #cv2.circle(img1, tuple(D[i]), 10, (255,255,255), -1)
                #cv2.circle(img2, tuple(E[i]), 10, (255,255,255), -1)
            B=np.array(B)
            rect = cv2.minAreaRect(B)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(img1, [box], -1, CL,10)
            C=np.array(C)
            rect = cv2.minAreaRect(C)
            box = np.int0(cv2.boxPoints(rect))
            cv2.drawContours(img2, [box], -1, CL, 10)
test1.abc(D,E,M,N)
#print(M)
test1.abcd(D,M,E,N)
#print(M,1111111111,N)
e0=[]
e1=[]
test1.Merge(M,N,e0,e1)
print(e0)
for i in range(len(e0)):
    CL = (random.randint(0,255),random.randint(0,255),random.randint(0,255)) 
    for p in range(len(e0[i])):
        cv2.circle(img1, tuple(e0[i][p]), 30, CL, -1)
        cv2.circle(img2, tuple(e1[i][p]), 30, CL, -1)
#h,w = img1.shape[:2]
#height = h
#width = 30
#image = np.zeros((height, width, 3), dtype=np.uint8)
#res = np.hstack((img1,image))
#matched = cv2.drawMatches(img1,kpts1,img2,kpts2,good,None,(0,255,0))
res = np.hstack((img1,img2))

cv2.imwrite("matched.png", res)

win = cv2.namedWindow('test win2', flags=0)

cv2.imshow('test win2', res)
cv2.waitKey()
cv2.destroyAllWindows()