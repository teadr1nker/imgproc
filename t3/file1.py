#!/usr/bin/python3
import numpy as np
import cv2

image = cv2.imread('forest.jpg')

grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imwrite('grayforest.jpg', grayimg)
