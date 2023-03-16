#!/usr/bin/python3
import cv2
import numpy as np

# contrast
image = cv2.imread('forest.jpg')
contrasted = cv2.addWeighted(image,
                             3, #contrast
                             image,
                             0,
                             1) #brightnes
cv2.imwrite('contrastedforest.jpg', contrasted)

# solarization
Xmax = image.max()
K = [.001, .002, .004, .008, .016]

lookUpTable = np.zeros((256, 1), dtype = 'uint8' )
for k in K:
    for x in range(256):
        lookUpTable[x] = k * x * (Xmax - x)
        #lookUpTable[x] = np.abs(np.sin(x * solarization_const)) * 100

    cv2.imwrite(f'solarizedK={k}.png',
                cv2.LUT(image, lookUpTable))
