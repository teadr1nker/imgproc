#!/usr/bin/python3
import cv2
import numpy as np

# lines
image = cv2.imread('images/lines.jpg')
imageG = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# make better
ret, thresh = cv2.threshold(imageG, 200, 255, cv2.THRESH_BINARY)
kernel = np.ones((5, 5))
# thresh = cv2.GaussianBlur(thresh, (3, 3), cv2.BORDER_DEFAULT)
edges = cv2.dilate(thresh, kernel)
kernel = np.ones((5, 5))
edges = cv2.erode(edges, kernel)
# edges = cv2.Canny(thresh, 100, 200, apertureSize=3)

# thresh = cv2.GaussianBlur(thresh, (5, 5), cv2.BORDER_DEFAULT)
# edges = cv2.dilate(edges, kernel)
lines = cv2.HoughLines(edges, 1, np.pi/360, 143)

if lines is not None:
    for i in range(0, len(lines)):
        rho = lines[i][0][0]
        theta = lines[i][0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
        pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
        cv2.line(image, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)

cv2.imwrite('images/linesHough.jpg', image)
