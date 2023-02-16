#!/usr/bin/python3
import cv2
import numpy as np

def rotate(image, degree, shift = (0, 0)):
    rads = np.radians(degree)


    rot_img = np.uint8(np.zeros(image.shape))

    height = rot_img.shape[0]
    width = rot_img.shape[1]

    midx, midy = (width//2, height//2)

    for i in range(height):
        for j in range(width):
            x = (i-midy) * np.cos(rads)+(j - midx) * np.sin(rads)
            y = -(i-midy)*np.sin(rads)+(j-midx)*np.cos(rads)

            x = round(x)+midy + shift[0]
            y = round(y)+midx + shift[1]

            if x >= 0 and y >= 0 and x < height and  y < width:
                rot_img[i, j, :] = image[x, y, :]

    return rot_img

image = cv2.imread('image.jpg')
print(f'Size: {image.shape} ')

# Image rotation
cv2.imwrite('rotated.jpg', rotate(image, 60, (128, 128)))

