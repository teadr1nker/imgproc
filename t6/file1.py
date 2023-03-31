#!/usr/bin/python3
import cv2
import numpy as np

# circles
kiwi = cv2.imread('../t2/kiwi.png')
kiwiG = cv2.cvtColor(kiwi, cv2.COLOR_BGR2GRAY)

# body
blur = cv2.GaussianBlur(kiwiG, (11, 11), 0)
ret, thresh = cv2.threshold(blur, 25, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
kiwiContures = cv2.drawContours(kiwi, contours, 21, (0, 255, 0), 3)

# eye
ret, thresh = cv2.threshold(kiwiG, 0, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
kiwiContures = cv2.drawContours(kiwi, contours, 2, (0, 255, 0), 3)

# squares
ret, thresh = cv2.threshold(kiwiG, 254, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
kiwiContures = cv2.drawContours(kiwi, contours, 0, (255, 20, 0), 3)

cv2.imwrite('images/kiwi.png', kiwiContures)

# Area and perimiter
textBG = contours[0]
area = cv2.contourArea(textBG)
perimeter = cv2.arcLength(textBG, True)
print(f'Area:{area}\nPerimiter{perimeter}')

