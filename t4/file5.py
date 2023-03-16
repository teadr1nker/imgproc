#!/usr/bin/python3
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('noise.png')[:, :, 0]

blur = cv2.GaussianBlur(image, (15, 15), 0)

_, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imwrite('otsuImage.jpg', th)
