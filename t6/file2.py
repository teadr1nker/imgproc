#!/usr/bin/python3
import cv2
import numpy as np

# star contour
star = cv2.imread('images/star.jpg')
starG = cv2.cvtColor(star, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(starG, 200, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour = contours[1]
hull = cv2.convexHull(contour)
result = cv2.drawContours(star, [hull], -1, (0, 0, 255), 5)
cv2.imwrite('images/starContour.jpg', result)

# deffects
hull = cv2.convexHull(contour, returnPoints=False)
defects = cv2.convexityDefects(contour, hull)

for i in range(defects.shape[0]):
    s, e, f, d = defects[i, 0]
    start = tuple(contour[s][0])
    end = tuple(contour[e][0])
    far = tuple(contour[f][0])
    cv2.line(star, start, end, [0, 255, 0], 3)  # Погрешности аппроксимации
    cv2.circle(star, far, 5, [0, 0, 255], -1)

cv2.imwrite("images/starDeffects.jpg", star)

# circle

(x, y), radius = cv2.minEnclosingCircle(contour)
center = (int(x), int(y))
radius = int(radius)
cv2.circle(star, center, radius, (0, 255, 0), 3)

# rectangle

rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(star, [box], 0, (0, 0, 255), 3)

# ellipse

ellipse = cv2.fitEllipse(contour)
cv2.ellipse(star, ellipse, (255, 0, 0), 2)

# axis
rows, cols = star.shape[:2]
[vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
cv2.line(star, (cols-1, righty), (0, lefty), (0, 255, 0), 2)


cv2.imwrite("images/starFigures.jpg", star)
