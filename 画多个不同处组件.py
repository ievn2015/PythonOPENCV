# -*- coding: utf-8 -*-
"""
Created on Thu May 24 01:02:52 2018

@author: Windows
"""

areas = np.zeros( len(cnts) )  
    idx = 0  
    for cont in cnts :
        areas[idx] = cv2.contourArea(cont)  
        idx = idx + 1  
    areas_s = cv2.sortIdx(areas, cv2.SORT_DESCENDING | cv2.SORT_EVERY_COLUMN)
    #c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
    (b8, g8, r8) = cv2.split(img1)  
  
# 对每个区域进行处理  
    draw_img1 = img1.copy()
    for idx in areas_s :  
        if areas[idx] < 100 :  
            break  
        rect = cv2.minAreaRect(areas[idx])
        box = np.int0(cv2.boxPoints(rect))

    #dist1 = np.linalg.norm(box[0] - box[-1])
    #print (dist1)
    #if dist1 <100:
    
        draw_img1 = cv2.drawContours(draw_img1, [box], -1, (255, 0, 0), 15)