#!/usr/bin/python3
import cv2 as cv
import numpy as np

# 2
image = cv.imread('images/windows.jpg')
sample = image[49:209, 80:234]
imageG = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
sampleG = cv.cvtColor(sample, cv.COLOR_BGR2GRAY)
# cv.imshow('sample', sample)
# cv.waitKey()

# 3.1
sift = cv.SIFT_create()

keyPointsSample, dstS = sift.detectAndCompute(sampleG, None)
keyPoints, dst = sift.detectAndCompute(imageG, None)

imageKP = cv.drawKeypoints(imageG, keyPoints, image)
sampleKP = cv.drawKeypoints(sampleG, keyPointsSample, sample)

print('image')
for point in keyPoints:
    print(f'coords: {point.pt} angle: {point.angle}')
print('sample')
for point in keyPointsSample:
    print(f'coords: {point.pt} angle: {point.angle}')

# 3.2
bf = cv.BFMatcher()
matches = bf.knnMatch(dstS, dst, k=2)

good = []
for m, n in matches:
    if m.distance < .75 * n.distance:
        good.append([m])

imageSIFT = cv.drawMatchesKnn(sample ,keyPointsSample,
                         image, keyPoints,
                         good, None,
                         flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# cv.imshow('matches', imageMatches)
# cv.waitKey()

# 3.3
FLANN_INDEX_KDTREE = 1
indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
searchParams = dict(checks=50)

flann = cv.FlannBasedMatcher(indexParams, searchParams)
matches = flann.knnMatch(dstS, dst, k=2)

matchesMask = [[0, 0] for i in range(len(matches))]

for i, (m, n) in enumerate(matches):
    if m.distance < .7 * n.distance:
        matchesMask[i]=[1, 0]

drawParams = dict(matchColor = (0 ,255 ,0),
                   singlePointColor = (255, 0, 0),
                   matchesMask = matchesMask,
                   flags = cv.DrawMatchesFlags_DEFAULT)

imageFLANN = cv.drawMatchesKnn(sample ,keyPointsSample,
                  image, keyPoints,
                  matches, None, **drawParams)

cv.imshow('matches', imageFLANN)
cv.waitKey()
