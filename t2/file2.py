#!/usr/bin/python3
import cv2
import numpy as np

image = cv2.imread('image.jpg')
rows, cols, ch = image.shape

#Affine transformationr
pts1 = np.float32([[50, 50],
                   [200, 50],
                   [50, 200]])

pts2 = np.float32([[10, 100],
                   [200, 50],
                   [100, 250]])

M = cv2.getAffineTransform(pts1, pts2)
affineImage = cv2.warpAffine(image, M, (cols, rows))
cv2.imwrite('affine.jpg', affineImage)

# Image projection
srcPoints = np.float32([[0, 0], [cols-1, 0],
                        [0, rows-1], [cols-1, rows-1]])
dstPoints = np.float32([[0, 0], [cols-1, int(rows * .33)],
                        [0, rows-1], [cols-1, int(rows * .66)]])

M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
projected = cv2.warpPerspective(image, M, (cols, rows))
cv2.imwrite('projected.jpg', projected)

# Image wrapping
wrapped = np.zeros(image.shape, dtype=image.dtype)
for i in range(rows):
    for j in range(cols):
        x = 0
        y = int(32. * np.sin(3.14 * j / 150))
        if i + y < rows:
            wrapped[i, j] = image[(i + y) % rows, j]
    else:
        wrapped[i, j] = 0

cv2.imwrite('wrapped.jpg', wrapped)
