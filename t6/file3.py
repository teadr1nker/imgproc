#!/usr/bin/python3
import numpy as np
import cv2

carriage = cv2.imread("../t4/railway_carriage.jpg")
carriageG = cv2.cvtColor(carriage, cv2.COLOR_BGR2GRAY)
mask = np.zeros(carriage.shape[:2], np.uint8)
print(carriage.shape)
mask[150:700, 135:610] = 255
masked = cv2.bitwise_and(carriageG, carriageG, mask = mask)
ret, thresh = cv2.threshold(masked, 128, 255, 0)
contours, hierarchy = cv2.findContours(thresh,
                                       mode=cv2.RETR_TREE,
                                       method=cv2.CHAIN_APPROX_NONE)

maxArea = .0
for i in range(len(contours)):
    area = cv2.contourArea(contours[i])
    if area > maxArea:
        maxArea = area
        contour = contours[i]

cv2.drawContours(carriage, contour,
                 -1, (0, 0, 255), 8)

cv2.imwrite('images/carriageContour.jpg', carriage)
