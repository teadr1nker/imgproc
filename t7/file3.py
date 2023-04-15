#!/usr/bin/python3
import cv2
import numpy as np

image = cv2.imread('images/square_many2.jpg')
imageG = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template = image[13:47, 290:325]

w = template.shape[0]
h = template.shape[1]

squares = cv2.matchTemplate(image,
                            template,
                            cv2.TM_CCOEFF_NORMED)
threshold = .93
loc = np.where(squares > threshold)

squareCount = 0

for pt in zip(*loc[::-1]):
    cv2.rectangle(image, pt,
                  (pt[0]+w, pt[1]+h),
                  (0, 0, 0), 2)
    squareCount += 1

print(squareCount)
cv2.imwrite('images/squares.jpg', image)
