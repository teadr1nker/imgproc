#!/usr/bin/python3
import cv2 as cv
import numpy as np

def harris(path):
    image = cv.imread(path)
    name = path.split('/')[-1]
    name = name.capitalize()
    path = '/'.join(path.split('/')[:-1])

    imageG = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corners = cv.cornerHarris(imageG, 8, 3, .04)
    image[corners > .05 * corners.max()] = [0, 0, 255]
    cv.imwrite(path + '/harris' + name, image)

def tomasi(path):
    image = cv.imread(path)
    name = path.split('/')[-1]
    name = name.capitalize()
    path = '/'.join(path.split('/')[:-1])

    imageG = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(imageG, 500, .05, 5)
    corners = np.intp(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv.circle(image, (x, y), 3, 255, -1)

    cv.imwrite(path + '/shiTomasi' + name, image)


images = ['images/home.jpg',
          'images/sudoku.png',
          'images/frame.jpg']

for i in images:
    harris(i)
    tomasi(i)
