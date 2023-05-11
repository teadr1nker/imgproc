#!/usr/bin/python3
import cv2 as cv
import numpy as np
from file1 import numbers
import shutil

def randomNoise(image, n):
    for i in range(n):
        x, y = (np.random.randint(0, image.shape[0]),
                np.random.randint(0, image.shape[1]))
        image[x, y] = [0, 0, 0]

pixels = [20, 50, 100, 200]

for number in numbers:
    for i in range(4):
        for j in range(20):
            image = cv.imread(f'images/numbers/test/{number}/{j+801}.jpg')
            randomNoise(image, pixels[i])
            cv.imwrite(f'images/numbers/test{i+2}/{number}/{j+801}.jpg', image)

# create validation
n = 800
val = .2
shift = 200
for number in numbers:
    for i in range(int(n * val)):
        name = f'{i + shift}.jpg'
        shutil.copyfile(f'images/numbers/train/{number}/{name}',
                        f'images/numbers/val/{number}/{name}')
