#!/usr/bin/python3
import cv2
import numpy as np

# blurring text
image = cv2.imread('text.png')

kernel = np.ones((5, 5)) / 25.

cv2.imwrite('blurrytext.png',
            cv2.filter2D(image, -1, kernel))

# bilateral filter

image = cv2.imread('forest.jpg')

cv2.imwrite('bilateralforest.jpg',
            cv2.bilateralFilter(image, 9, 75, 75))
