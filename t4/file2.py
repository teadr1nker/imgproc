#!/usr/bin/python3
import cv2
import numpy as np
import matplotlib.pyplot as plt

city = cv2.imread('city.jpg')
grayCity = cv2.cvtColor(city, cv2.COLOR_BGR2GRAY)
cv2.imwrite('grayCity.jpg', grayCity)
eq = cv2.equalizeHist(grayCity)
cv2.imwrite('eqcity.jpg', eq)
plt.plot(cv2.calcHist([grayCity], [0], None, [256], [0, 256]))
plt.plot(cv2.calcHist([eq], [0], None, [256], [0, 256]))
plt.savefig('cityHistComp.png')
