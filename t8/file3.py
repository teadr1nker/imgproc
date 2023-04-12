#!/usr/bin/python3
import cv2 as cv
import numpy as np

windows = cv.imread('images/windows.jpg')
sample = windows[49:209, 80:234]
sampleG = cv.cvtColor(sample, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
sampleKP, sampleDst = sift.detectAndCompute(sampleG ,None)

# a function to match them all
def findMatches(image, suffix='a'):
    imageG = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    imageKP, imageDst = sift.detectAndCompute(imageG ,None)

    FLANN_INDEX_KDTREE = 1
    indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    searchParams = dict(checks = 50)

    flann = cv.FlannBasedMatcher(indexParams,searchParams)

    matches = flann.knnMatch(sampleDst, imageDst, k=2)

    matchesMask = [[0, 0] for i in range(len(matches))]

    for i,(m, n) in enumerate(matches):
        if m.distance < .75 * n.distance:
            matchesMask[i] = [1, 0]

    drawParams = dict(matchColor = (0, 255, 0),
                   singlePointColor = (255, 0, 0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

    imageMatches = cv.drawMatchesKnn(sample, sampleKP,
                                     image , imageKP,
                                     matches, None, **drawParams)

    cv.imshow('matches ' + suffix, imageMatches)
    cv.waitKey()

# image transformations
windows90 = cv.rotate(windows, cv.ROTATE_90_CLOCKWISE)
findMatches(windows90)

mtx = cv.getRotationMatrix2D((windows.shape[1] / 2, windows.shape[0] / 2), 210, 0.6)
windows210 = cv.warpAffine(windows, mtx, (windows.shape[1], windows.shape[0]))
findMatches(windows210, 'b')

mtx = cv.getRotationMatrix2D((windows.shape[1] / 2, windows.shape[0] / 2), 0, 2)
windows2x = cv.warpAffine(windows, mtx, (windows.shape[1], windows.shape[0]))
findMatches(windows2x, 'c')

mtx = cv.getRotationMatrix2D((windows.shape[1] / 2, windows.shape[0] / 2), 0, .5)
windows05x = cv.warpAffine(windows, mtx, (windows.shape[1], windows.shape[0]))
findMatches(windows05x, 'd')

mtx = cv.getRotationMatrix2D((windows.shape[1] / 2, windows.shape[0] / 2), 270, 1.5)
windows27005x = cv.warpAffine(windows, mtx, (windows.shape[1], windows.shape[0]))
findMatches(windows27005x, 'e')

