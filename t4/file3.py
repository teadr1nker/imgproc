#!/usr/bin/python3
import cv2
import matplotlib.pyplot as plt

types =[cv2.THRESH_BINARY,
        cv2.THRESH_BINARY_INV,
        cv2.THRESH_TRUNC,
        cv2.THRESH_TOZERO,
        cv2.THRESH_TOZERO_INV]

titles = ['Original',
          'BINARY',
          'BINARY_INV',
          'TRUNC',
          'TOZERO',
          'TOZERO_INV']

gray = cv2.imread('../t3/grayforest.jpg')

images = [gray]

for t in types:
    images.append(cv2.threshold(gray, 127, 255, t)[1])

for i, image in enumerate(images):
    plt.subplot(2,3,i+1);plt.imshow(image, 'gray')
    plt.title(titles[i])
    plt.xticks([]);plt.yticks([])

plt.savefig('threshold.jpg')
plt.show()
