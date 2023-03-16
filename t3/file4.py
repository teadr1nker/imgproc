#!/usr/bin/python3
import cv2

gray = cv2.imread('grayforest.jpg')

# Sobel
cv2.imwrite('SobelV.jpg',
            cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
cv2.imwrite('SobelH.jpg',
            cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))

# Laplican

cv2.imwrite('Laplass.jpg',
            cv2.Laplacian(gray, cv2.CV_64F))

# Canny
cv2.imwrite('Canny.jpg,',
            cv2.Canny(gray, 200, 240))
