import cv2
import numpy as np
import matplotlib.pyplot as plt

 
# 使用2g-r-b分离土壤与背景
 
src = cv2.imread('1.jpg')#imread是计算机语言中的一个函数，用于读取图片文件中的数据。
#print(src)
#cv2.imshow('src', src)
 
# 转换为浮点数进行计算
fsrc = np.array(src, dtype=np.float32) /255.0
(b,g,r) = cv2.split(fsrc)
gray =2*b-g-r
 
# 求取最大值和最小值
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray)
  
# 计算直方图
#hist = cv2.calcHist([gray], [0], None, [256], [minVal, maxVal])
#plt.plot(hist)
#plt.show()
 
#cv2.waitKey()
# 转换为u8类型，进行otsu二值化
gray_u8 = np.array((gray - minVal) / (maxVal - minVal) * 255, dtype=np.uint8)
(thresh, bin_img) = cv2.threshold(gray_u8, -1.0, 255, cv2.THRESH_OTSU)
cv2.imshow('bin_img', bin_img)
_a, cnts, _b= cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# import pdb
# pdb.set_trace()
d = sorted(cnts, key=cv2.contourArea, reverse=True)
areas = np.zeros( len(d) )
idx = 0
for cont in d : 
    areas[idx] = cv2.contourArea(cont)
    print(areas[idx])
    idx = idx + 1
for i in range(len(areas)):
    if areas[i] < 100:
        break
    c=d[i]
    #print(len(c))
# OpenCV中通过cv2.drawContours在图像上绘制轮廓。
# 第一个参数是指明在哪幅图像上绘制轮廓
# 第二个参数是轮廓本身，在Python中是一个list
# 第三个参数指定绘制轮廓list中的哪条轮廓，如果是-1，则绘制其中的所有轮廓
# 第四个参数是轮廓线条的颜色
# 第五个参数是轮廓线条的粗细
 
# cv2.minAreaRect()函数:
# 主要求得包含点集最小面积的矩形，这个矩形是可以有偏转角度的，可以与图像的边界不平行。
# compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)

# rect = cv2.minAreaRect(cnts[1])
    box = np.int0(cv2.boxPoints(rect))
 
 
# draw a bounding box arounded the detected barcode and display the image
    cv2.drawContours(src, [box], -1, (0, 255, 0), 3)
    cv2.imshow("Image", src)
    cv2.imwrite("contoursImage2.jpg", src)
#cv2.waitKey(0)
 
# step7：裁剪。box里保存的是绿色矩形区域四个顶点的坐标。我将按下图红色矩形所示裁剪昆虫图像。
# 找出四个顶点的x，y坐标的最大最小值。新图像的高=maxY-minY，宽=maxX-minX。
#Xs = [i[0] for i in box]
#Ys = [i[1] for i in box]
#x1 = min(Xs)
#x2 = max(Xs)
#y2 = max(Ys)
#hight = y2 - y1
#width = x2 - x1
#cropImg = src[y1:y1+hight, x1:x1+width]
 
# show image
#cv2.imshow("cropImg", cropImg)
#cv2.imwrite("bee.jpg", cropImg)
cv2.waitKey()
cv2.destroyAllWindows()