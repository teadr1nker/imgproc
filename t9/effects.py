import cv2 as cv
import numpy as np

def noEffect(image):
    return image

def blur3x3(image):
    return cv.filter2D(image, -1,
                       np.ones((3, 3), np.float32) / 9)

def blur5x5(image):
    return cv.filter2D(image, -1,
                       np.ones((5, 5), np.float32) / 25)

def blur9x9(image):
    return cv.filter2D(image, -1,
                       np.ones((9, 9), np.float32) / 81)

def blur15x15(image):
    return cv.filter2D(image, -1,
                       np.ones((15, 15), np.float32) / 225)

def removeB(image):
    image[:, :, 0] = 0
    return image

def removeG(image):
    image[:, :, 1] = 0
    return image

def removeR(image):
    image[:, :, 2] = 0
    return image

def erode3x3(image):
    return cv.erode(image, np.ones((3, 3), np.uint8))

def erode5x5(image):
    return cv.erode(image, np.ones((5, 5), np.uint8))

effects = [noEffect,
           blur3x3,
           blur5x5,
           blur9x9,
           blur15x15,
           removeB,
           removeG,
           removeR,
           erode3x3,
           erode5x5]
