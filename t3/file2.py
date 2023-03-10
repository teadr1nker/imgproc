#!/usr/bin/python3
import cv2

image = cv2.imread('forest.jpg')
contrasted = cv2.addWeighted(image,
                             3, #contrast
                             image,
                             0,
                             1) #brightnes
cv2.imwrite('contrastedforest.jpg', contrasted)

