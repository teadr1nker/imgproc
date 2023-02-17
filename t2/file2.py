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
cv2.imwrite('affine1.jpg', affineImage)

pts1 = np.float32([[50, 50],
                   [200, 50],
                   [50, 200]])

pts2 = np.float32([[60, 60],
                   [50, 200],
                   [200, 50]])

M = cv2.getAffineTransform(pts1, pts2)
affineImage = cv2.warpAffine(image, M, (cols, rows))
cv2.imwrite('affine2.jpg', affineImage)


# Image projection
srcPoints = np.float32([[0, 0], [cols-1, 0],
                        [0, rows-1], [cols-1, rows-1]])
dstPoints = np.float32([[0, 0], [cols-1, int(rows * .33)],
                        [0, rows-1], [cols-1, int(rows * .66)]])

M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
projected = cv2.warpPerspective(image, M, (cols, rows))
cv2.imwrite('projected1.jpg', projected)

srcPoints = np.float32([[0, 0], [cols-1, 0],
                        [0, rows-1], [cols-1, rows-1]])
dstPoints = np.float32([[0, 0], [cols-1, 0],
                        [cols * .2, rows-1], [cols * .8, rows-1]])

M = cv2.getPerspectiveTransform(srcPoints, dstPoints)
projected = cv2.warpPerspective(image, M, (cols, rows))
cv2.imwrite('projected2.jpg', projected)

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

cv2.imwrite('wrapped1.jpg', wrapped)

for i in range(rows):
    for j in range(cols):
        x = int(32. * np.sin(3.14 * j / 64))
        y = int(32. * np.sin(2 * 3.14 * j / 128))
        if j + x < cols and i + y < rows:
            wrapped[i, j] = image[(i + y) % rows, (j + x) % cols]
    else:
        wrapped[i, j] = 0

cv2.imwrite('wrapped2.jpg', wrapped)
