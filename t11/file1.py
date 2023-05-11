#!/usr/bin/python3
import cv2 as cv
import numpy as np

numbers = ['0', '1', '3', '8']

whiteBox = np.zeros((100, 100, 3), np.uint8)
whiteBox.fill(255)

for number in numbers:
    for i in range(820):
        coords = [np.random.randint(10, 55),
                  np.random.randint(60, 98)]
        image = np.copy(whiteBox)
        image = cv.putText(image,
                   number,
                   coords,
                   cv.FONT_HERSHEY_SIMPLEX,
                   2.4,
                   [0, 0, 0],
                   2,
                   cv.LINE_AA)
        if i < 400:
            image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        elif i >=810:
            image = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)

        if i < 800:
            cv.imwrite(f'images/numbers/train/{number}/{i+1}.jpg', image)
        else:
            cv.imwrite(f'images/numbers/test/{number}/{i+1}.jpg', image)

