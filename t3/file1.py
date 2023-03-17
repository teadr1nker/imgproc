#!/usr/bin/python3
import numpy as np
import cv2

def contrast(img, dif):
    Xmax = img.max()
    Xmin = img.min()
    Ymin = (Xmax - Xmin)/2 - dif // 2
    Ymax = (Xmax - Xmin)/2 + dif // 2

    res = np.zeros(img.shape, np.uint8)
    for i, row in enumerate(img):
        for j, x in enumerate(row):
            res[i, j] = (((x - Xmin) / (Xmax - Xmin)) * (Ymax - Ymin)) + Ymin

    rev = np.zeros(img.shape, np.uint8)
    for i, row in enumerate(res):
        for j, y in enumerate(row):
            rev[i, j] = (((y - Ymin) / (Ymax - Ymin)) * (Xmax - Xmin)) + Xmin

    return res, rev

image = cv2.imread('forest.jpg')

grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('grayforest.jpg', grayimg)

contrasted, reverse = contrast(grayimg, 50)

cv2.imwrite('grayforestcontrast.png', contrasted)
cv2.imwrite('grayforestreversed.png', reverse)

