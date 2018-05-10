# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 00:01:19 2018

@author: Administrator
"""


import cv2
import numpy as np

def match_image(Image, Target, value):
    import cv2
    import numpy as np
    #加载原始RGB图像
    img_rgb = cv2.imread(Image)
    #创建一个原始图像的灰度版本，所有操作在灰度版本中处理，然后在RGB图像中使用相同坐标还原
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #加载将要匹配的图像模板
    template = cv2.imread(Target, 0)
    #记录图像模板的尺寸
    w, h = template.shape[::-1]
    #使用matchTemplate对原始灰度图像和图像模板进行匹配
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    #阈值设定
    threshold = value
    #res大于90%
    loc = np.where(res >= threshold)
    if len(loc[0]):
        print("匹配目标人物成功")
        #使用灰度图像中的坐标对原始RGB图像进行标记
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (7, 249, 151), 2)
        #显示图像
        cv2.imshow('Detected', img_rgb)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True
    else:
        print("匹配目标人物失败")
        return False

if __name__ == '__main__':
    Target = '1.jpg'
    Image = "2.jpg"
    value = 0.9
    match_image(Image, Target, value)
    