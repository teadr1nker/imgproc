#!/usr/bin/python3
import cv2
import matplotlib.pyplot as plt

gray = cv2.imread('railway_carriage.jpg')[:, :, 0]
images = [gray,
cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                      cv2.THRESH_BINARY, 11, 2),
cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                      cv2.THRESH_BINARY_INV, 11, 2),
cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                      cv2.THRESH_BINARY, 11, 2),
cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                      cv2.THRESH_BINARY_INV, 11, 2),
cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                      cv2.THRESH_BINARY_INV, 7, 3)
         ]

titles = ['Original',
          'mean binary 11',
          'mean binary_inv 11',
          'gaussian binary 11',
          'gaussian binary_inv 11',
          'mean binary_inv 7']

for i, image in enumerate(images):
    plt.subplot(2,3,i+1);plt.imshow(image, 'gray')
    plt.title(titles[i])
    plt.xticks([]);plt.yticks([])

plt.savefig('thresholdAdaptive.jpg')
plt.show()

