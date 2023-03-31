#!/usr/bin/python3
import cv2
import numpy as np

def processing(path, kernel):
    image = cv2.imread(path)
    name = path.split('/')[-1]
    name = name.capitalize()
    print(name)
    path = '/'.join(path.split('/')[:-1])

    # Erosion

    cv2.imwrite(path + '/erosion' + name,
                cv2.erode(image, kernel, iterations = 1))

    # Dialation
    cv2.imwrite(path + '/dialation' + name,
                cv2.dilate(image, kernel, iterations = 1))

    # Opening
    cv2.imwrite(path + '/opening' + name,
                cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel))

    # Closing
    cv2.imwrite(path + '/closing' + name,
                cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel))

    # Gradient
    cv2.imwrite(path + '/gradient' + name,
                cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel))

    # Top hat
    cv2.imwrite(path + '/topHat' + name,
                cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel))

    # Black hat
    cv2.imwrite(path + '/blackhat' + name,
                cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel))


images = ['images/forest.jpg',
          'images/sudoku.png',
          'images/at.jpg']

kernels = [np.ones((11, 11), np.uint8),
           np.ones((5, 5), np.uint8),
           np.ones((7, 7), np.uint8)]

for i in range(3):
    processing(images[i], kernels[i])
