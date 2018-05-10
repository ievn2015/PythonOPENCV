# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:49:35 2018
字符画生成方案, 运行下面的程序会打开一个图片预览窗口显示生成的字符画图像
@author: Windows
"""

import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageStat

font = PIL.ImageFont.truetype('consola', 14)

im = PIL.Image.open('d:/me.jpg')
im = im.convert('F')
size = im.size

rx = im.size[0]
ry = int(rx / size[0] * size[1] * 8 / 14)
im = im.resize((rx, ry), PIL.Image.NEAREST)

mean = PIL.ImageStat.Stat(im).mean[0]

words = []
for y in range(im.size[1]):
    for x in range(im.size[0]):
        p = im.getpixel((x, y))
        if p < mean / 2:
            c = '#'
        elif mean / 2 <= p < mean:
            c = '='
        elif mean <= p < mean + (255 - mean) / 2:
            c = '-'
        elif mean + (255 - mean) / 2 <= p:
            c = ' '
        else:
            raise ValueError(p)
        words.append(c)
    words.append('\n')

im.close()

im = PIL.Image.new('RGB', (im.size[0] * 8, im.size[1] * 14), '#FFFFFF')
dr = PIL.ImageDraw.Draw(im)
dr.text((0, 0), ''.join(words), '#000000', font)
im = im.resize(size, PIL.Image.LANCZOS)
im.show()