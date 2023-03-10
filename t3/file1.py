#!/usr/bin/python3
#import numpy as np
import cv2

def contrast(img, dif):
    Xmax = img.max()
    Xmin = img.min()
    Ymin = (Xmax - Xmin)/2 - dif // 2
    Ymax = (Xmax - Xmin)/2 + dif // 2

    res = img
    for i, row in enumerate(img):
        for j, x in enumerate(row):
            res[i, j] = (((x - Xmin) / (Xmax - Xmin)) * (Ymax - Ymin)) + Ymin
    return res

image = cv2.imread('forest.jpg')

grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('grayforest.jpg', grayimg)

cv2.imwrite('grayforestcontrast.jpg', contrast(grayimg, 50))

