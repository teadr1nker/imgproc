#!/usr/bin/python3
import cv2
# import numpy as np

def bgr2yuv(image):
    newImage = image
    B = image[:, :, 0]
    G = image[:, :, 1]
    R = image[:, :, 2]

    # Y U V
    newImage[:, :, 0] = .299 * R + .587 * G + .114 * B
    newImage[:, :, 1] = -.14713 * R - .28886 * G + .436 * B + 128
    newImage[:, :, 2] = .615 * R - .51499 * G - .10001 * B + 128

    # print(newImage)
    return newImage

image = cv2.imread('image.jpg')
print(f'Size: {image.shape} ')
yuvImage = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
cv2.imwrite('yuvimage.jpg', yuvImage)
cv2.imwrite('myyuvimage.jpg', bgr2yuv(image))
yuvImage[:, :, 0] *= 2
yuvImage[:, :, 1] *= 3
yuvImage[:, :, 2] //= 2
cv2.imwrite('yuvimage2.jpg', yuvImage)
cv2.imwrite('bgrimage1.jpg', cv2.cvtColor(yuvImage, cv2.COLOR_YUV2BGR))

hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imwrite('hsvimage.jpg', hsvImage)
hsvImage[:, :, 0] *= 2
hsvImage[:, :, 1] //= 2
hsvImage[:, :, 2] *= 4
cv2.imwrite('hsvimage2.jpg', hsvImage)
cv2.imwrite('bgrimage2.jpg', cv2.cvtColor(hsvImage, cv2.COLOR_HSV2BGR))
