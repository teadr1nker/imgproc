#!/usr/bin/python3
import cv2
import numpy as np

def processing(path):
    image = cv2.imread(path)
    name = path.split('/')[-1]
    path = '/'.join(path.split('/')[:-1])
    # print(path, name)

    kernel = np.ones((5, 5), np.uint8)
    # Erosion

    cv2.imwrite(path + '/erosion' + name,
                cv2.erode(image, kernel, iterations = 2))

    # Dialation
    cv2.imwrite(path + '/dialation' + name,
                cv2.dilate(image, kernel, iterations = 2))

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
          'images/sudoku.png']

for i in images:
    processing(i)
