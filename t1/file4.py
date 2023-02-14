#!/usr/bin/python3
import cv2
import numpy as np

def brg2yuv(image):
    newImage = image
    B = image[:, :, 0]
    R = image[:, :, 1]
    G = image[:, :, 2]

    # Y U V
    newImage[:, :, 0] = .299 * R + .587 * G + .114 * B
    newImage[:, :, 1] = -.14713 * R - .28886 * G + .436 * B + 128
    newImage[:, :, 2] = .615 * R - .51499 * G - .10001 * B + 128

    #print(newImage)
    return newImage

image = cv2.imread('image.jpg')
print(f'Size: {image.shape} ')

yuvimg = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
cv2.imwrite('yuvimage.jpg', yuvimg)
cv2.imwrite('yuvimage2.jpg', brg2yuv(image))
cv2.imwrite('hsvimage.jpg', cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
#cv2.imwrite('hsvimage2.jpg', brg2hsv(image))
