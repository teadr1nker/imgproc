#!/usr/bin/python3
import cv2
import numpy as np

image = cv2.imread('image.jpg')
w, h, ch = image.shape
image = cv2.rectangle(image,
                     [0, 0],
                     [130, 64],
                     [255, 255, 255],
                     -1)
# Body
image = cv2.ellipse(image,
                    [h//2, w//2],
                    [int(h*0.3), int(w*0.3)],
                    0,
                    0,
                    360,
                    [12, 107, 12],
                    -1)
# Head
image = cv2.circle(image,
                   [1400, 333],
                   200,
                   [10, 97, 10],
                   -1)
# Eye
image = cv2.circle(image,
                   [1430, 250],
                   10,
                   [0, 0, 0],
                   -1)

# Beek
pts = np.array([[1550, 300], [1550, 350], [1750, 325]], np.int32)
image = cv2.fillPoly(image,
                      [pts],
                      [8, 30, 53],
                      )
image = cv2.putText(image,
                    'Kiwi',
                    [16, 50],
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    [255, 0, 0],
                    4,
                    cv2.LINE_AA)

cv2.imwrite('kiwi.png', image)
