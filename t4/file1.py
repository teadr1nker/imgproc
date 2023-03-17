#!/usr/bin/python3
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('../t3/forest.jpg')
grayimg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

grayHist = cv2.calcHist([image], [0], None, [256], [0,256])

plt.plot(grayHist)
plt.savefig('grayHist.jpg')
plt.clf()

h, w, _ = image.shape
mask = np.zeros((h, w), np.uint8)
mask[h//2:h, 0:w] = 255

plt.plot(cv2.calcHist([image], [0], mask, [256], [0, 256]), color="blue")
plt.plot(cv2.calcHist([image], [1], mask, [256], [0, 256]), color="red")
plt.plot(cv2.calcHist([image], [2], mask, [256], [0, 256]), color="green")
plt.savefig('maskedHistBRG.jpg')
plt.clf()
