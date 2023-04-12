#!/usr/bin/python3
import cv2
import numpy as np

image = cv2.imread('images/skyskeb_one.jpg')
imageG = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel = np.ones((9, 9), np.float32)

tower = cv2.filter2D(imageG, -1, kernel / 81)
tower = cv2.Canny(tower, 50, 150)

lines = cv2.HoughLines(tower, 2, np.pi, 90)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 500*(-b)), int(y0 + 500*(a)))
        pt2 = (int(x0 - 500*(-b)), int(y0 - 500*(a)))
        cv2.line(image, pt1, pt2, (0, 255, 0), 4, cv2.LINE_AA)

cv2.imwrite('images/towerEdges.jpg', image)
