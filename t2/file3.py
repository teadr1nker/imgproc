#!/usr/bin/python3
import cv2
import numpy as np
from file1 import rotate

image = cv2.imread('image.jpg')
rows, cols, ch = image.shape

experiment = image
midy, midx = rows//2, cols//2
# 1st quadrant
experiment[:midy, :midx , 1] = 0
# 2nd quadrant
experiment[:midy, midx:] = rotate(experiment[:midy, midx:], 180)
# 3d quadrant
experiment[midy:, :midx] = experiment[midy:, :midx]//2 + experiment[midy:, midx:]//2
# 4th quadrant
w, h, = 60, 120

experiment[midy + 16: midy + h + 16, midx + 16: midx + 16 + w] = \
    experiment[0: h, 0: w]

experiment[midy + 16: midy + h + 16, midx + 96: midx + 96 + w] = \
    experiment[0: h, midx: midx + w]

experiment[midy + 16: midy + h + 16, midx + 172: midx + 172 + w] = \
    experiment[midy: midy + h, 0: w]

# Frame
thickness = 16
color = np.array([0, 0, 255], dtype=image.dtype)
framed = np.zeros((rows + thickness*2, cols + thickness*2, 3), dtype=experiment.dtype)
for i in range(rows + thickness*2):
    for j in range(cols + thickness*2):
        if i < thickness or i >= rows + thickness or j < thickness or j >= cols + thickness:
            framed[i, j] = color
        else:
            framed[i, j] = experiment[i - thickness, j - thickness]


cv2.imwrite('experiment.jpg', framed)
