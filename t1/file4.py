#!/usr/bin/python3
import cv2

def brg2yuv(image):
    newImage = image
    B = image[:, :, 0]
    R = image[:, :, 1]
    G = image[:, :, 2]

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
# cv2.imshow('yuv image', yuvImage)
cv2.waitKey()
cv2.imwrite('Y yuv.jpg', yuvImage[:, :, 0])
cv2.imwrite('U yuv.jpg', yuvImage[:, :, 1])
cv2.imwrite('V yuv.jpg', yuvImage[:, :, 2])
cv2.waitKey()
cv2.imwrite('myyuvimage.jpg', brg2yuv(image))
hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imwrite('hsvimage.jpg', hsvImage)
cv2.imwrite('H hsv.jpg', yuvImage[:, :, 0])
cv2.imwrite('S hsv.jpg', yuvImage[:, :, 1])
cv2.imwrite('V hsv.jpg', yuvImage[:, :, 2])
# cv2.imwrite('hsvimage2.jpg', brg2hsv(image))
